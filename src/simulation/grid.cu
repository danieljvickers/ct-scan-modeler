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
        grid.periodic[d] = is_active && parameters.boundary[d][0] == bc_periodic;
        grid.lower_boundary[d] = !grid.periodic[d];
        grid.upper_boundary[d] = !grid.periodic[d];
    }

    // slabs along the slowest axis, each needs at least a halo of cells
    const int axis = grid.decomposed_axis;
    grid.cells[axis] = partCount(grid.global_cells[axis], num_ranks, rank);
    grid.offset[axis] = partOffset(grid.global_cells[axis], num_ranks, rank);
    grid.lower_boundary[axis] = !grid.periodic[axis] && rank == 0;
    grid.upper_boundary[axis] = !grid.periodic[axis] && rank == num_ranks - 1;
    CTM_REQUIRE(grid.cells[axis] >= halo_width, "%d ranks leave fewer than %d cells per slab along axis %d, use fewer GPUs",
                num_ranks, halo_width, axis);

    for (int d = 0; d < max_dims; ++d) grid.storage[d] = grid.cells[d] + 2*grid.halo[d];
    return grid;
}

void tileCounts(const GridView& box, const int shift[max_dims], int tiles[max_dims]) {
    for (int d = 0; d < max_dims; ++d) tiles[d] = (box.storage[d] + (1 << shift[d]) - 1) >> shift[d];
}

GridView TileSet::build(const GridView& box, const int shift[max_dims], const std::vector<char>& active_tiles) {
    GridView grid = box;
    grid.tiled = true;
    for (int d = 0; d < max_dims; ++d) grid.tile_shift[d] = shift[d];
    tileCounts(box, shift, grid.tiles);
    const long total = static_cast<long>(grid.tiles[0])*grid.tiles[1]*grid.tiles[2];
    CTM_REQUIRE(static_cast<long>(active_tiles.size()) == total, "tile flags cover %ld tiles, the box has %ld", static_cast<long>(active_tiles.size()), total);

    this->host_tile_slots.assign(total, -1);
    this->host_slot_tiles.clear();
    for (long tile = 0; tile < total; ++tile) {
        if (!active_tiles[tile]) continue;
        this->host_tile_slots[tile] = static_cast<int>(this->host_slot_tiles.size());
        this->host_slot_tiles.push_back(static_cast<int>(tile));
    }
    this->tile_slots.copyFromHost(this->host_tile_slots.data(), total);
    this->slot_tiles.copyFromHost(this->host_slot_tiles.data(), static_cast<long>(this->host_slot_tiles.size()));
    grid.num_tiles = static_cast<long>(this->host_slot_tiles.size());
    grid.tile_slots = this->tile_slots.data();
    grid.slot_tiles = this->slot_tiles.data();
    grid.host_tile_slots = this->host_tile_slots.data();
    grid.host_slot_tiles = this->host_slot_tiles.data();
    return grid;
}
