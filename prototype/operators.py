"""Second order finite volume operators on a collocated grid, stored as sparse matrices.

Matrices are assembled on the host with SciPy and copied to the backend for time stepping.
"""

import functools
import operator
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as host_sparse

from prototype.backend import sparse_to_device, to_device
from prototype.boundary_conditions import Halo

DIRECTIONS = ("x", "y")


def sum_over_directions(terms):
    """Add per-direction terms without a leading 0, which CuPy sparse matrices reject."""
    return functools.reduce(operator.add, terms)


@dataclass
class FaceOperator:
    """Affine maps from cell values to face values and face normal gradients along one direction."""
    interpolation: object  # sparse (faces, cells)
    interpolation_offset: object
    gradient: object
    gradient_offset: object

    def to_device(self):
        return replace(self, interpolation=sparse_to_device(self.interpolation), interpolation_offset=to_device(self.interpolation_offset),
                       gradient=sparse_to_device(self.gradient), gradient_offset=to_device(self.gradient_offset))

    def interpolate(self, cell_values):
        return self.interpolation @ cell_values + self.interpolation_offset

    def face_gradient(self, cell_values):
        return self.gradient @ cell_values + self.gradient_offset


def _face_shape(grid, axis):
    return tuple(size + (dim == axis) for dim, size in enumerate(grid.shape))


def _axis_geometry(grid, axis):
    if axis == 0:
        return grid.x_cc, grid.x_cb, grid.dx
    return grid.y_cc, grid.y_cb, grid.dy


def _face_operator(grid, axis, halo_beg, halo_end):
    centers, boundaries, widths = _axis_geometry(grid, axis)
    num_axis_cells = grid.shape[axis]
    face_shape = _face_shape(grid, axis)
    num_faces = int(np.prod(face_shape))
    face_index = np.arange(num_faces).reshape(face_shape)
    axis_index = np.indices(face_shape)[axis]

    def cell_index(face_mask, offset):
        multi_index = list(np.indices(face_shape)[:, face_mask])
        multi_index[axis] = multi_index[axis] + offset
        return np.ravel_multi_index(multi_index, grid.shape)

    rows, columns, interpolation_weights, gradient_weights = [], [], [], []
    interpolation_offset = np.zeros(num_faces)
    gradient_offset = np.zeros(num_faces)

    # interior faces sit between cells axis_index - 1 and axis_index
    interior = (axis_index > 0) & (axis_index < num_axis_cells)
    right = axis_index[interior]
    distance = centers[right] - centers[right - 1]
    right_weight = (boundaries[right] - centers[right - 1])/distance
    for offset, interpolation_weight, gradient_weight in ((-1, 1.0 - right_weight, -1.0/distance), (0, right_weight, 1.0/distance)):
        rows.append(face_index[interior])
        columns.append(cell_index(interior, offset))
        interpolation_weights.append(interpolation_weight)
        gradient_weights.append(gradient_weight)

    # boundary faces close the stencil with the halo relation
    for face_axis_index, cell_axis_index, halo, outward_sign in ((0, 0, halo_beg, -1.0), (num_axis_cells, num_axis_cells - 1, halo_end, 1.0)):
        face_mask = axis_index == face_axis_index
        width = widths[cell_axis_index]
        rows.append(face_index[face_mask])
        columns.append(cell_index(face_mask, cell_axis_index - face_axis_index))
        interpolation_weights.append(np.full(face_mask.sum(), 0.5*(1.0 + halo.coefficient)))
        gradient_weights.append(np.full(face_mask.sum(), outward_sign*(halo.coefficient - 1.0)/width))
        interpolation_offset[face_index[face_mask]] = 0.5*halo.value
        gradient_offset[face_index[face_mask]] = outward_sign*halo.value/width

    rows, columns = np.concatenate(rows), np.concatenate(columns)
    shape = (num_faces, grid.num_cells)
    interpolation = host_sparse.csr_matrix((np.concatenate(interpolation_weights), (rows, columns)), shape=shape)
    gradient = host_sparse.csr_matrix((np.concatenate(gradient_weights), (rows, columns)), shape=shape)
    return FaceOperator(interpolation, interpolation_offset, gradient, gradient_offset)


def _divergence(grid, axis):
    """Cell average of the face flux difference along one direction: (F_upper - F_lower)/width."""
    _, _, widths = _axis_geometry(grid, axis)
    face_shape = _face_shape(grid, axis)
    cell_multi_index = np.indices(grid.shape).reshape(2, -1)
    cells = np.ravel_multi_index(cell_multi_index, grid.shape)
    inverse_width = 1.0/widths[cell_multi_index[axis]]
    upper_index = cell_multi_index.copy()
    upper_index[axis] += 1
    lower_faces = np.ravel_multi_index(cell_multi_index, face_shape)
    upper_faces = np.ravel_multi_index(upper_index, face_shape)
    return host_sparse.csr_matrix((np.concatenate([inverse_width, -inverse_width]),
                                   (np.concatenate([cells, cells]), np.concatenate([upper_faces, lower_faces]))),
                                  shape=(grid.num_cells, int(np.prod(face_shape))))


class Operators:
    """Face and cell operators for every variable, built once from the grid and boundary conditions."""

    def __init__(self, grid, halos):
        self.host_divergence = {direction: _divergence(grid, axis) for axis, direction in enumerate(DIRECTIONS)}
        self.host_face = {variable: {direction: _face_operator(grid, axis, halos[variable][(direction, "beg")],
                                                          halos[variable][(direction, "end")])
                                for axis, direction in enumerate(DIRECTIONS)}
                     for variable in halos}

        # the pressure correction shares the pressure halos with homogeneous values
        homogeneous = {side: Halo(halo.coefficient, 0.0) for side, halo in halos["p"].items()}
        self.host_face["phi"] = {direction: _face_operator(grid, axis, homogeneous[(direction, "beg")], homogeneous[(direction, "end")])
                            for axis, direction in enumerate(DIRECTIONS)}

        # zero gradient interpolation for cell centered pressure gradients
        neumann = Halo(1.0, 0.0)
        self.host_face["neumann"] = {direction: _face_operator(grid, axis, neumann, neumann) for axis, direction in enumerate(DIRECTIONS)}

        # backend copies used every time step
        self.divergence = {direction: sparse_to_device(matrix) for direction, matrix in self.host_divergence.items()}
        self.face = {variable: {direction: face.to_device() for direction, face in faces.items()} for variable, faces in self.host_face.items()}

    def host_laplacian(self, variable):
        """Return the host matrix and offset of the cell centered Laplacian, for assembling implicit systems."""
        matrix = sum_over_directions(self.host_divergence[d] @ self.host_face[variable][d].gradient for d in DIRECTIONS)
        offset = sum_over_directions(self.host_divergence[d] @ self.host_face[variable][d].gradient_offset for d in DIRECTIONS)
        return matrix.tocsr(), offset

    def laplacian(self, variable):
        """Return the backend matrix and offset of the cell centered Laplacian."""
        matrix, offset = self.host_laplacian(variable)
        return sparse_to_device(matrix), to_device(offset)

    def cell_gradient(self, variable, cell_values, direction):
        """Cell centered gradient from interpolated face values."""
        return self.divergence[direction] @ self.face[variable][direction].interpolate(cell_values)

    def convection(self, variable, cell_values, face_velocities):
        """Conservative convection div(U q) with divergence free face velocities and central face values."""
        return sum_over_directions(self.divergence[d] @ (face_velocities[d]*self.face[variable][d].interpolate(cell_values)) for d in DIRECTIONS)

    def face_divergence(self, face_velocities):
        return sum_over_directions(self.divergence[d] @ face_velocities[d] for d in DIRECTIONS)
