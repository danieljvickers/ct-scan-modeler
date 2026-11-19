"""Linear solvers for the implicit velocity and pressure systems.

DirectSolver:   sparse LU of the full immersed boundary system.
SpectralSolver: exact solve of shift*I + scale*Laplacian on a uniform grid with sine and cosine transforms.
KrylovSolver:   right preconditioned restarted GMRES, Saad & Schultz (1986), with a SpectralSolver preconditioner.
"""

import numpy as np

from prototype.backend import fft, sparse_linalg, sparse_to_device, to_device, to_host, xp

# halo coefficient pair (begin, end) -> transform type, +1 is zero gradient and -1 is zero value at the face
NEUMANN, DIRICHLET = 1.0, -1.0
TRANSFORM_TYPES = {(NEUMANN, NEUMANN): "dct2", (DIRICHLET, DIRICHLET): "dst2",
                   (NEUMANN, DIRICHLET): "dct4", (DIRICHLET, NEUMANN): "dst4"}
WAVE_SHIFTS = {"dct2": 0.0, "dst2": 1.0, "dct4": 0.5, "dst4": 0.5}  # wave k has angle pi*(k + shift)/N


def _transform_4(values, axis, sine):
    """Orthonormal DCT-IV or DST-IV through a zero padded complex FFT, both are their own inverse."""
    num_points = values.shape[axis]
    shape = [1]*values.ndim
    shape[axis] = num_points
    index = xp.arange(num_points).reshape(shape)
    pre_twiddle = xp.exp(-1j*np.pi*index/(2*num_points))
    post_twiddle = xp.exp(-1j*np.pi*(2*index + 1)/(4*num_points))
    spectrum = xp.fft.fft(values*pre_twiddle, n=2*num_points, axis=axis)
    spectrum = xp.take(spectrum, xp.arange(num_points), axis=axis)*post_twiddle
    result = -spectrum.imag if sine else spectrum.real
    return result*np.sqrt(2.0/num_points)


def forward_transform(values, axis, transform_type):
    if transform_type == "dct2":
        return fft.dct(values, type=2, axis=axis, norm="ortho")
    if transform_type == "dst2":
        return fft.dst(values, type=2, axis=axis, norm="ortho")
    return _transform_4(values, axis, sine=transform_type == "dst4")


def inverse_transform(values, axis, transform_type):
    if transform_type == "dct2":
        return fft.idct(values, type=2, axis=axis, norm="ortho")
    if transform_type == "dst2":
        return fft.idst(values, type=2, axis=axis, norm="ortho")
    return _transform_4(values, axis, sine=transform_type == "dst4")


def laplacian_eigenvalues(num_points, spacing, transform_type):
    """Eigenvalues of the 1D cell-centered second difference for the waves of this transform."""
    wave_angles = np.pi*(np.arange(num_points) + WAVE_SHIFTS[transform_type])/(2*num_points)
    return -(4.0/spacing**2)*np.sin(wave_angles)**2


class DirectSolver:

    def __init__(self, host_matrix):
        self.factorization = sparse_linalg.splu(sparse_to_device(host_matrix).tocsc())
        self.last_iterations = 0

    def solve(self, rhs, initial_guess=None):
        return self.factorization.solve(rhs)


class SpectralSolver:
    """Exact solve of (shift*I + scale*L) x = b where L is the uniform grid Laplacian with domain halos only."""

    def __init__(self, grid, halos, shift, scale):
        spacings = (grid.dx, grid.dy)
        for spacing in spacings:
            assert np.allclose(spacing, spacing[0]), "ERROR :: the spectral solver needs a uniform grid"
        self.shape = grid.shape
        self.transform_types = [TRANSFORM_TYPES[(halos[(direction, "beg")].coefficient, halos[(direction, "end")].coefficient)]
                                for direction in ("x", "y")]
        eigenvalues_x, eigenvalues_y = (laplacian_eigenvalues(num, spacing[0], kind)
                                        for num, spacing, kind in zip(self.shape, spacings, self.transform_types))
        eigenvalues = shift + scale*(eigenvalues_x[:, None] + eigenvalues_y[None, :])
        assert np.abs(eigenvalues).min() > 0, "ERROR :: the operator is singular, add an outflow boundary"
        self.inverse_eigenvalues = to_device(1.0/eigenvalues)

    def solve(self, rhs, initial_guess=None):
        values = rhs.reshape(self.shape)
        for axis, kind in enumerate(self.transform_types):
            values = forward_transform(values, axis, kind)
        values = values*self.inverse_eigenvalues
        for axis, kind in reversed(list(enumerate(self.transform_types))):
            values = inverse_transform(values, axis, kind)
        return values.ravel()


def gmres(matrix, rhs, preconditioner, initial_guess, relative_tolerance, restart, max_iterations):
    """Right preconditioned restarted GMRES, returns the solution and the iteration count."""
    tolerance = relative_tolerance*float(xp.linalg.norm(rhs))
    solution = xp.zeros_like(rhs) if initial_guess is None else initial_guess.copy()
    if tolerance == 0.0:
        return xp.zeros_like(rhs), 0

    iterations = 0
    while True:
        residual = rhs - matrix @ solution
        residual_norm = float(xp.linalg.norm(residual))
        if residual_norm <= tolerance:
            return solution, iterations
        assert iterations < max_iterations, f"ERROR :: GMRES stalled at relative residual {residual_norm*relative_tolerance/tolerance:.2e}"

        # build the Krylov basis with classical Gram-Schmidt applied twice
        basis = xp.empty((restart + 1, rhs.size))
        basis[0] = residual/residual_norm
        hessenberg = np.zeros((restart + 1, restart))
        for column in range(restart):
            vector = matrix @ preconditioner.solve(basis[column])
            for _ in range(2):
                projections = basis[:column + 1] @ vector
                vector -= projections @ basis[:column + 1]
                hessenberg[:column + 1, column] += to_host(projections)
            hessenberg[column + 1, column] = float(xp.linalg.norm(vector))
            iterations += 1

            # small least squares problem on the host gives the residual estimate
            target = np.zeros(column + 2)
            target[0] = residual_norm
            coefficients = np.linalg.lstsq(hessenberg[:column + 2, :column + 1], target, rcond=None)[0]
            estimate = np.linalg.norm(target - hessenberg[:column + 2, :column + 1] @ coefficients)
            breakdown = hessenberg[column + 1, column] <= 1e-14*residual_norm
            if estimate <= tolerance or breakdown or iterations >= max_iterations:
                break
            basis[column + 1] = vector/hessenberg[column + 1, column]
        solution += preconditioner.solve(to_device(coefficients) @ basis[:column + 1])


class KrylovSolver:
    """GMRES on the immersed boundary system, preconditioned by the exact no-body spectral solve."""

    def __init__(self, host_matrix, preconditioner, relative_tolerance, restart, max_iterations):
        self.matrix = sparse_to_device(host_matrix)
        self.preconditioner = preconditioner
        self.relative_tolerance = relative_tolerance
        self.restart = restart
        self.max_iterations = max_iterations
        self.last_iterations = 0

    def solve(self, rhs, initial_guess=None):
        solution, self.last_iterations = gmres(self.matrix, rhs, self.preconditioner, initial_guess,
                                               self.relative_tolerance, self.restart, self.max_iterations)
        return solution
