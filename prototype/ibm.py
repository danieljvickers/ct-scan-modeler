"""Ghost cell immersed boundary method, following MFC's m_ibm.

Ghost-cell IBM: Tseng & Ferziger (2003), Mittal et al. (2008).
"""

import numpy as np
import scipy.sparse as host_sparse

from prototype.backend import sparse_to_device, to_device, xp

STENCIL_OFFSETS = ((0, 0), (1, 0), (0, 1), (1, 1))  # 2x2 image point stencil from its lower corner
DISTANCE_EPS = 1.0e-12  # relative to the cell width, below this a point sits on a node


class GhostPoints:
    """Struct of arrays describing every ghost point and its image point."""

    def __init__(self, cells, patch_ids, levelsets, normals, image_points, stencil_cells, interp_coeffs, is_bilinear):
        self.cells = cells                  # flattened ghost cell index
        self.patch_ids = patch_ids          # immersed boundary patch owning the ghost point
        self.levelsets = levelsets          # distance from the ghost cell center to the surface
        self.normals = normals              # outward unit surface normal
        self.image_points = image_points    # mirror of the ghost cell center across the surface
        self.stencil_cells = stencil_cells  # (num_gps, 4) flattened cells interpolated to the image point
        self.interp_coeffs = interp_coeffs  # (num_gps, 4) interpolation weights, zero on solid cells
        self.is_bilinear = is_bilinear      # false where the stencil touched a solid and fell back to IDW

    def __len__(self):
        return len(self.cells)


def find_ghost_cells(ib_markers, gp_layers):
    """A solid cell with any fluid cell within gp_layers cells in each direction is a ghost point."""
    padded = np.pad(ib_markers, gp_layers, mode="edge")
    has_fluid_neighbor = np.zeros(ib_markers.shape, dtype=bool)
    for shift_i in range(2*gp_layers + 1):
        for shift_j in range(2*gp_layers + 1):
            has_fluid_neighbor |= padded[shift_i:shift_i + ib_markers.shape[0], shift_j:shift_j + ib_markers.shape[1]] == 0
    return (ib_markers != 0) & has_fluid_neighbor


def interpolation_weights(grid, image_points, lower_i, lower_j, is_fluid):
    """Bilinear weights on all-fluid stencils, else inverse distance squared on the fluid cells, Chaudhuri et al. (2011)."""
    stencil_x = grid.x_cc[np.stack([lower_i + di for di, _ in STENCIL_OFFSETS], axis=1)]
    stencil_y = grid.y_cc[np.stack([lower_j + dj for _, dj in STENCIL_OFFSETS], axis=1)]

    # bilinear weights, Mittal et al. (2008)
    fraction_x = np.clip((image_points[:, 0] - grid.x_cc[lower_i])/(grid.x_cc[lower_i + 1] - grid.x_cc[lower_i]), 0.0, 1.0)
    fraction_y = np.clip((image_points[:, 1] - grid.y_cc[lower_j])/(grid.y_cc[lower_j + 1] - grid.y_cc[lower_j]), 0.0, 1.0)
    bilinear = np.column_stack([(1 - fraction_x)*(1 - fraction_y), fraction_x*(1 - fraction_y),
                                (1 - fraction_x)*fraction_y, fraction_x*fraction_y])

    # inverse distance squared, exact copy when the image point lands on a cell center
    distances = np.hypot(stencil_x - image_points[:, [0]], stencil_y - image_points[:, [1]])
    on_node = distances <= DISTANCE_EPS*grid.min_cell_width
    inverse_distance = np.where(on_node.any(axis=1, keepdims=True), on_node.astype(float),
                                1.0/np.maximum(distances, DISTANCE_EPS*grid.min_cell_width)**2)

    is_bilinear = is_fluid.all(axis=1)
    weights = np.where(is_bilinear[:, None], bilinear, inverse_distance)*is_fluid
    weight_sums = weights.sum(axis=1, keepdims=True)
    assert (weight_sums > 0).all(), "ERROR :: an image point stencil holds no fluid cells, refine the grid near the body"
    return weights/weight_sums, is_bilinear


class ImmersedBoundaries:
    """Ghost points, cell classification, and the ghost point relations used by the solver."""

    def __init__(self, case, grid, ib_markers, models):
        self.grid = grid
        self.ib_markers = ib_markers
        self.num_ibs = case["num_ibs"]
        ghost_mask = find_ghost_cells(ib_markers, case["gp_layers"])

        # ghost point geometry is set up on the host, the solver uses the device copies below
        self.ghost_points = self._compute_ghost_points(case, ghost_mask, models)
        self.host_image_point_matrix = self._image_point_matrix()
        self.host_cell_types = {"fluid": (ib_markers == 0).ravel(), "ghost": ghost_mask.ravel()}
        self.host_cell_types["interior"] = ~self.host_cell_types["fluid"] & ~self.host_cell_types["ghost"]

        self.image_point_matrix = sparse_to_device(self.host_image_point_matrix)
        self.is_fluid, self.is_ghost, self.is_interior = (to_device(self.host_cell_types[name]) for name in ("fluid", "ghost", "interior"))
        self.markers = to_device(ib_markers.ravel())
        self.cell_volumes = to_device(grid.cell_volumes.ravel())

    def _compute_ghost_points(self, case, ghost_mask, models):
        grid = self.grid
        ghost_i, ghost_j = np.nonzero(ghost_mask)
        cells = np.ravel_multi_index((ghost_i, ghost_j), grid.shape)
        patch_ids = self.ib_markers[ghost_i, ghost_j]
        centers = np.column_stack([grid.x_cc[ghost_i], grid.y_cc[ghost_j]])

        levelsets = np.empty(len(cells))
        normals = np.empty((len(cells), 2))
        for patch_id in range(1, self.num_ibs + 1):
            owned = patch_ids == patch_id
            model = models[patch_id - 1]
            levelsets[owned], normals[owned] = model.levelset(centers[owned])

        # image point mirrors the ghost point across the surface
        image_points = centers + 2.0*levelsets[:, None]*normals

        # lower corner of the cell-center stencil that brackets the image point
        lower_i = np.searchsorted(grid.x_cc, image_points[:, 0]) - 1
        lower_j = np.searchsorted(grid.y_cc, image_points[:, 1]) - 1
        assert ((lower_i >= 0) & (lower_i < grid.m) & (lower_j >= 0) & (lower_j < grid.n)).all(), \
            "ERROR :: an image point lies outside the domain, move the body away from the boundary"
        stencil_cells = np.stack([np.ravel_multi_index((lower_i + di, lower_j + dj), grid.shape) for di, dj in STENCIL_OFFSETS], axis=1)
        interp_coeffs, is_bilinear = interpolation_weights(grid, image_points, lower_i, lower_j, (self.ib_markers.ravel() == 0)[stencil_cells])
        return GhostPoints(cells, patch_ids, levelsets, normals, image_points, stencil_cells, interp_coeffs, is_bilinear)

    def _image_point_matrix(self):
        """Sparse map from cell values to image point values, nonzero only on ghost cell rows."""
        ghost_points = self.ghost_points
        rows = np.repeat(ghost_points.cells, len(STENCIL_OFFSETS))
        return host_sparse.csr_matrix((ghost_points.interp_coeffs.ravel(), (rows, ghost_points.stencil_cells.ravel())),
                                      shape=(self.grid.num_cells, self.grid.num_cells))

    def replace_solid_rows(self, base_matrix, ghost_sign):
        """Impose q_ghost + ghost_sign*q_image = 0 on ghost rows, on the host.

        Ghost rows are scaled by the base diagonal and interior rows keep the base operator, so the system
        differs from the no-body operator only on ghost rows, which keeps the spectral preconditioner effective.
        """
        identity = host_sparse.identity(self.grid.num_cells, format="csr")
        ghost_scaling = host_sparse.diags(self.host_cell_types["ghost"]*base_matrix.diagonal())
        is_not_ghost = host_sparse.diags((~self.host_cell_types["ghost"]).astype(float))
        return (is_not_ghost @ base_matrix + ghost_scaling @ (identity + ghost_sign*self.host_image_point_matrix)).tocsr()

    def apply_ghost_velocity(self, velocity):
        """Mirror the image point velocity into each ghost cell for a stationary no-slip wall."""
        velocity[self.is_ghost] = -(self.image_point_matrix @ velocity)[self.is_ghost]
        velocity[self.is_interior] = 0.0

    def compute_forces(self, operators, pressure, velocities, laplacians, kinematic_viscosity):
        """Body forces per unit density from the stress divergence summed over solid cells, as in MFC."""
        forces = xp.zeros((self.num_ibs, 2))
        for component, (direction, variable) in enumerate((("x", "u"), ("y", "v"))):
            matrix, offset = laplacians[variable]
            stress_divergence = (-operators.cell_gradient("p", pressure, direction)
                                 + kinematic_viscosity*(matrix @ velocities[variable] + offset))
            forces[:, component] = xp.bincount(self.markers, weights=stress_divergence*self.cell_volumes, minlength=self.num_ibs + 1)[1:]
        return forces
