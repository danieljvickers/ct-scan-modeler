"""Incremental pressure correction on a collocated grid with face velocities.

Projection: Kim & Moin (1985) with the incremental pressure form of Van Kan (1986).
Collocated face velocities: Zang, Street & Koseff (1994), used with ghost cells by Ye et al. (1999).
"""

import numpy as np
import scipy.sparse as host_sparse

from toolchain import constants
from prototype.backend import to_device, xp
from prototype.linear_solvers import DirectSolver, KrylovSolver, SpectralSolver
from prototype.operators import DIRECTIONS, sum_over_directions

VELOCITIES = ("u", "v")


class FractionalStepSolver:
    """AB2 convection, Crank-Nicolson diffusion, and an IB aware pressure Poisson solve."""

    def __init__(self, case, grid, operators, immersed_boundaries, halos):
        self.case = case
        self.grid = grid
        self.operators = operators
        self.ib = immersed_boundaries
        self.halos = halos
        self.dt = case["dt"]
        self.kinematic_viscosity = 1.0/case["fluid_pp(1)%Re(1)"] if case["viscous"] else 0.0
        self.laplacians = {variable: operators.laplacian(variable) for variable in VELOCITIES}
        self.previous_convection = None

        # velocity Helmholtz systems, built once since dt and viscosity are fixed
        identity = host_sparse.identity(grid.num_cells, format="csr")
        diffusion_number = 0.5*self.dt*self.kinematic_viscosity
        self.helmholtz_solvers = {variable: self._build_solver(identity - diffusion_number*operators.host_laplacian(variable)[0],
                                                               variable, shift=1.0, scale=-diffusion_number, ghost_sign=1.0)
                                  for variable in VELOCITIES}

        # pressure Poisson system with a zero normal pressure gradient mirrored into the ghost cells
        poisson_matrix = sum_over_directions(operators.host_divergence[d] @ operators.host_face["phi"][d].gradient for d in DIRECTIONS)
        self.poisson_solver = self._build_solver(poisson_matrix, "p", shift=0.0, scale=1.0, ghost_sign=-1.0)
        self.pressure_correction = None

        # Rhie-Chow style face pressure smoothing only on interior faces between two fluid cells
        self.fluid_faces = {}
        is_solid = np.zeros(grid.num_cells) if self.ib is None else (~self.ib.host_cell_types["fluid"]).astype(float)
        for d in DIRECTIONS:
            neighbors = (operators.host_face["neumann"][d].interpolation != 0).astype(float)
            is_interior_face = np.diff(neighbors.indptr) == 2
            self.fluid_faces[d] = to_device(is_interior_face & (neighbors @ is_solid == 0))

    def _build_solver(self, host_matrix, variable, shift, scale, ghost_sign):
        """Apply the immersed boundary rows on the host, then set up the solver chosen by poisson_solver."""
        if self.ib is not None:
            host_matrix = self.ib.replace_solid_rows(host_matrix, ghost_sign)
        if self.case["poisson_solver"] == constants.POISSON_SPARSE_LU:
            return DirectSolver(host_matrix)
        preconditioner = SpectralSolver(self.grid, self.halos[variable], shift, scale)
        return KrylovSolver(host_matrix, preconditioner, self.case["krylov_tolerance"], self.case["krylov_restart"],
                            self.case["krylov_max_iterations"])

    def last_iterations(self):
        """Krylov iterations of the last u, v, and pressure solves."""
        return [self.helmholtz_solvers[variable].last_iterations for variable in VELOCITIES] + [self.poisson_solver.last_iterations]

    def _project(self, predicted_velocities, predicted_face_velocities):
        """Solve for the pressure correction and make the face velocities divergence free."""
        rhs = self.operators.face_divergence(predicted_face_velocities)/self.dt
        if self.ib is not None:
            rhs[~self.ib.is_fluid] = 0.0
        pressure_correction = self.poisson_solver.solve(rhs, initial_guess=self.pressure_correction)
        self.pressure_correction = pressure_correction

        face_velocities = {d: predicted_face_velocities[d] - self.dt*self.operators.face["phi"][d].face_gradient(pressure_correction) for d in DIRECTIONS}
        velocities = {variable: predicted_velocities[variable] - self.dt*self.operators.cell_gradient("phi", pressure_correction, d)
                      for variable, d in zip(VELOCITIES, DIRECTIONS)}
        if self.ib is not None:
            for variable in VELOCITIES:
                self.ib.apply_ghost_velocity(velocities[variable])
        return velocities, face_velocities, pressure_correction

    def initialize(self, velocities, pressure):
        """Apply the immersed boundaries and project the initial field onto a divergence free one."""
        if self.ib is not None:
            for variable in VELOCITIES:
                self.ib.apply_ghost_velocity(velocities[variable])
        face_velocities = {d: self.operators.face[variable][d].interpolate(velocities[variable]) for variable, d in zip(VELOCITIES, DIRECTIONS)}
        velocities, face_velocities, _ = self._project(velocities, face_velocities)
        return velocities, face_velocities, pressure

    def step(self, velocities, face_velocities, pressure):
        """Advance one time step and return the new velocities, face velocities, and pressure."""
        dt, operators = self.dt, self.operators

        # explicit convection, Euler on the first step and AB2 after
        convection = {variable: operators.convection(variable, velocities[variable], face_velocities) for variable in VELOCITIES}
        previous_convection = convection if self.previous_convection is None else self.previous_convection
        self.previous_convection = convection

        # get tentative velocity with the old pressure gradient
        predicted = {}
        for variable, d in zip(VELOCITIES, DIRECTIONS):
            laplacian, laplacian_offset = self.laplacians[variable]
            rhs = (velocities[variable] - dt*(1.5*convection[variable] - 0.5*previous_convection[variable])
                   - dt*operators.cell_gradient("p", pressure, d)
                   + 0.5*dt*self.kinematic_viscosity*(laplacian @ velocities[variable]) + dt*self.kinematic_viscosity*laplacian_offset)
            if self.ib is not None:
                rhs[~self.ib.is_fluid] = 0.0  # stationary no-slip wall
            predicted[variable] = self.helmholtz_solvers[variable].solve(rhs, initial_guess=velocities[variable])

        # face velocities with the cell pressure gradient swapped for the compact face gradient
        predicted_faces = {}
        for variable, d in zip(VELOCITIES, DIRECTIONS):
            pressure_smoothing = (operators.face["neumann"][d].interpolate(operators.cell_gradient("p", pressure, d))
                                  - operators.face["p"][d].face_gradient(pressure))
            predicted_faces[d] = operators.face[variable][d].interpolate(predicted[variable]) + dt*self.fluid_faces[d]*pressure_smoothing

        velocities, face_velocities, pressure_correction = self._project(predicted, predicted_faces)
        return velocities, face_velocities, pressure + pressure_correction

    def max_fluid_divergence(self, face_velocities):
        divergence = self.operators.face_divergence(face_velocities)
        return float(xp.abs(divergence if self.ib is None else divergence[self.ib.is_fluid]).max())

    def forces(self, velocities, pressure):
        return self.ib.compute_forces(self.operators, pressure, velocities, self.laplacians, self.kinematic_viscosity)
