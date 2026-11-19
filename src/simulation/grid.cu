#include "grid.hpp"

GridView makeGrid(const Parameters& parameters) {
    GridView grid;
    grid.num_dims = parameters.num_dims;
    grid.decomposed_axis = parameters.num_dims - 1;
    const int rank = mpiRank();
    const int num_ranks = mpiSize();

    for (int d = 0; d < max_dims; ++d) {
        const bool is_active = d < parameters.num_dims;
        grid.global_cells[d] = parameters.global_cells[d];
        grid.origin[d] = parameters.domain_begin[d];
        grid.spacing[d] = (parameters.domain_end[d] - parameters.domain_begin[d])/parameters.global_cells[d];
        grid.halo[d] = is_active ? halo_width : 0;
        grid.cells[d] = grid.global_cells[d];
        grid.offset[d] = 0;
        grid.lower_boundary[d] = true;
        grid.upper_boundary[d] = true;
    }

    // slabs along the slowest axis, each needs at least a halo of cells
    const int axis = grid.decomposed_axis;
    grid.cells[axis] = partCount(grid.global_cells[axis], num_ranks, rank);
    grid.offset[axis] = partOffset(grid.global_cells[axis], num_ranks, rank);
    grid.lower_boundary[axis] = rank == 0;
    grid.upper_boundary[axis] = rank == num_ranks - 1;
    CTM_REQUIRE(grid.cells[axis] >= halo_width, "%d ranks leave fewer than %d cells per slab along axis %d, use fewer GPUs",
                num_ranks, halo_width, axis);

    for (int d = 0; d < max_dims; ++d) grid.storage[d] = grid.cells[d] + 2*grid.halo[d];
    return grid;
}
