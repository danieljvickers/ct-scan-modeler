"""Unit tests for the 2D prototype geometry, interpolation, and operators."""

import numpy as np

from prototype import model
from prototype.boundary_conditions import NEUMANN, dirichlet
from prototype.backend import to_host
from prototype.ibm import find_ghost_cells, interpolation_weights
from prototype.operators import Operators


class GridStub:
    """Uniform grid with only the fields the operators and interpolation read."""

    def __init__(self, num_x, num_y, length_x=1.0, length_y=1.0):
        self.m, self.n = num_x - 1, num_y - 1
        self.shape = (num_x, num_y)
        self.num_cells = num_x*num_y
        self.x_cb, self.y_cb = np.linspace(0, length_x, num_x + 1), np.linspace(0, length_y, num_y + 1)
        self.x_cc, self.y_cc = 0.5*(self.x_cb[1:] + self.x_cb[:-1]), 0.5*(self.y_cb[1:] + self.y_cb[:-1])
        self.dx, self.dy = np.diff(self.x_cb), np.diff(self.y_cb)
        self.min_cell_width = min(self.dx.min(), self.dy.min())


def disk_triangles(num_segments=64, radius=0.5):
    angles = np.linspace(0, 2*np.pi, num_segments + 1)
    rim = np.column_stack([radius*np.cos(angles), radius*np.sin(angles), np.zeros(num_segments + 1)])
    return np.stack([np.zeros((num_segments, 3)), rim[:-1], rim[1:]], axis=1)


def test_stl_ascii_and_binary_match(tmp_path):
    triangles = disk_triangles()
    ascii_path, binary_path = tmp_path/"disk_ascii.stl", tmp_path/"disk_binary.stl"
    with open(ascii_path, "w") as stl_file:
        stl_file.write("solid disk\n")
        for triangle in triangles:
            stl_file.write("facet normal 0 0 1\nouter loop\n" + "".join(f"vertex {x:.17e} {y:.17e} {z:.17e}\n" for x, y, z in triangle) + "endloop\nendfacet\n")
        stl_file.write("endsolid disk\n")
    record = np.zeros(len(triangles), dtype=[("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    record["vertices"] = triangles
    with open(binary_path, "wb") as stl_file:
        stl_file.write(b"\0"*80 + np.uint32(len(triangles)).tobytes() + record.tobytes())
    assert np.allclose(model.read_stl(ascii_path), triangles)
    assert np.allclose(model.read_stl(binary_path), triangles, atol=1e-7)


def test_disk_outline_winding_and_levelset():
    edges = model.boundary_edges_2d(disk_triangles(num_segments=64))
    assert len(edges) == 64  # interior spokes are shared and dropped
    winding = model.winding_number_2d(edges, np.array([[0.0, 0.0], [0.3, 0.1], [0.6, 0.0], [2.0, 2.0]]))
    assert np.allclose(winding, [1, 1, 0, 0], atol=1e-12)

    points = np.array([[0.2, 0.0], [0.0, -0.3]])
    distances, normals = model.boundary_distances_and_normals_2d(edges, points, tie_tolerance=1e-12)
    assert np.allclose(distances, [0.3, 0.2], atol=1e-3)
    assert np.allclose(normals, [[1.0, 0.0], [0.0, -1.0]], atol=1e-2)

    # points on a diagonal are equidistant from two edges and must get the mirrored normal
    diagonal = np.array([[0.3, 0.3], [0.3, -0.3]])
    _, diagonal_normals = model.boundary_distances_and_normals_2d(edges, diagonal, tie_tolerance=1e-12)
    assert np.allclose(diagonal_normals[0]*[1, -1], diagonal_normals[1], atol=1e-14)


def test_bilinear_reproduces_linear_fields_and_idw_fallback():
    grid = GridStub(8, 8)
    image_points = np.array([[0.31, 0.47], [0.52, 0.18]])
    lower_i = np.searchsorted(grid.x_cc, image_points[:, 0]) - 1
    lower_j = np.searchsorted(grid.y_cc, image_points[:, 1]) - 1
    stencil_x = grid.x_cc[np.stack([lower_i, lower_i + 1, lower_i, lower_i + 1], axis=1)]
    stencil_y = grid.y_cc[np.stack([lower_j, lower_j, lower_j + 1, lower_j + 1], axis=1)]
    linear_field = lambda x, y: 2.0 + 3.0*x - 5.0*y

    weights, is_bilinear = interpolation_weights(grid, image_points, lower_i, lower_j, np.ones((2, 4), dtype=bool))
    assert is_bilinear.all()
    assert np.allclose((weights*linear_field(stencil_x, stencil_y)).sum(axis=1), linear_field(*image_points.T))

    # a solid corner switches to inverse distance on the remaining fluid cells
    is_fluid = np.array([[True, True, True, False], [True, True, True, True]])
    weights, is_bilinear = interpolation_weights(grid, image_points, lower_i, lower_j, is_fluid)
    assert list(is_bilinear) == [False, True]
    assert weights[0, 3] == 0.0 and np.allclose(weights.sum(axis=1), 1.0)


def test_ghost_cells_border_the_fluid():
    markers = np.zeros((7, 7), dtype=np.int32)
    markers[2:5, 2:5] = 1
    ghost = find_ghost_cells(markers, gp_layers=1)
    assert ghost.sum() == 8 and not ghost[3, 3]


def test_laplacian_is_exact_for_quadratics():
    grid = GridStub(16, 12, length_x=2.0, length_y=1.5)
    halos = {"u": {(d, s): NEUMANN for d in "xy" for s in ("beg", "end")}}
    operators = Operators(grid, {**halos, "v": halos["u"], "p": halos["u"]})
    x_cc, y_cc = np.meshgrid(grid.x_cc, grid.y_cc, indexing="ij")
    matrix, _ = operators.laplacian("u")
    laplacian = to_host(matrix @ (x_cc**2 + 3*y_cc**2).ravel())
    interior = np.zeros(grid.shape, dtype=bool)
    interior[1:-1, 1:-1] = True
    assert np.allclose(laplacian.reshape(grid.shape)[interior], 8.0)


def test_dirichlet_halo_hits_face_value():
    grid = GridStub(4, 4)
    halo = dirichlet(2.5)
    halos = {"u": {(d, s): halo for d in "xy" for s in ("beg", "end")}}
    operators = Operators(grid, {**halos, "v": halos["u"], "p": halos["u"]})
    face_values = to_host(operators.face["u"]["x"].interpolate(np.full(grid.num_cells, 7.0)))
    boundary_faces = np.r_[0:grid.shape[1], grid.num_cells:grid.num_cells + grid.shape[1]]
    assert np.allclose(face_values[boundary_faces], 2.5)


def test_type_4_transforms_match_scipy():
    import scipy.fft
    from prototype.linear_solvers import forward_transform
    values = np.random.default_rng(0).random((12, 7))
    for kind, reference in (("dct4", scipy.fft.dct), ("dst4", scipy.fft.dst)):
        for axis in (0, 1):
            ours = to_host(forward_transform(values, axis, kind))
            assert np.allclose(ours, reference(values, type=4, axis=axis, norm="ortho"))
            assert np.allclose(to_host(forward_transform(ours, axis, kind)), values)  # self inverse


def test_spectral_solver_matches_sparse_solve_for_every_boundary_pair():
    import scipy.sparse.linalg
    from prototype.linear_solvers import SpectralSolver
    grid = GridStub(10, 6, length_x=2.0, length_y=1.2)
    rhs = np.random.default_rng(1).random(grid.num_cells)
    pairs = ((NEUMANN, dirichlet(0.0)), (dirichlet(0.0), NEUMANN), (dirichlet(0.0), dirichlet(0.0)))
    for (x_beg, x_end) in pairs:
        halos = {("x", "beg"): x_beg, ("x", "end"): x_end, ("y", "beg"): NEUMANN, ("y", "end"): NEUMANN}
        operators = Operators(grid, {"u": halos, "v": halos, "p": halos})
        matrix = operators.host_laplacian("u")[0]
        for shift, scale in ((0.0, 1.0), (1.0, -0.01)):
            system = shift*scipy.sparse.identity(grid.num_cells) + scale*matrix
            expected = scipy.sparse.linalg.spsolve(system.tocsc(), rhs)
            assert np.allclose(to_host(SpectralSolver(grid, halos, shift, scale).solve(rhs)), expected)
