#include "boundary_conditions.hpp"

namespace {

constexpr real neumann = 1.0;     // halo copies the interior, zero gradient
constexpr real dirichlet = -1.0;  // halo mirrors with opposite sign, zero value at the face

void setSide(HaloRule& rule, int direction, int side, real coefficient, real face_value) {
    rule.coefficient[direction][side] = coefficient;
    rule.value[direction][side] = coefficient == dirichlet ? 2.0*face_value : 0.0;  // linear through the face value
}

// fill the halo layers of one side along one axis, over the whole storage extent of the other axes
__global__ void fillSideKernel(GridView grid, real* field, int direction, int side, real coefficient, real value, const int* solid_markers) {
    const int other_1 = (direction + 1) % max_dims, other_2 = (direction + 2) % max_dims;
    const long plane = static_cast<long>(grid.storage[other_1])*grid.storage[other_2];
    const long thread = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (thread >= plane*grid.halo[direction]) return;
    const int layer = static_cast<int>(thread/plane);
    const long within = thread % plane;

    int halo_index[max_dims], interior_index[max_dims];
    halo_index[other_1] = interior_index[other_1] = static_cast<int>(within % grid.storage[other_1]) - grid.halo[other_1];
    halo_index[other_2] = interior_index[other_2] = static_cast<int>(within/grid.storage[other_1]) - grid.halo[other_2];
    const int cells = grid.cells[direction];
    halo_index[direction] = side == 0 ? -1 - layer : cells + layer;
    interior_index[direction] = side == 0 ? layer : cells - 1 - layer;  // mirror across the boundary face

    // a solid or ghost cell on the face takes no boundary value, the immersed boundary sets it
    const long interior_cell = grid.storageIndex(interior_index[0], interior_index[1], interior_index[2]);
    const real applied_value = solid_markers != nullptr && solid_markers[interior_cell] != 0 ? 0.0 : value;
    field[grid.storageIndex(halo_index[0], halo_index[1], halo_index[2])] = coefficient*field[interior_cell] + applied_value;
}

// wrap the halo layers of a periodic axis held entirely by this rank
__global__ void wrapSidesKernel(GridView grid, real* field, int direction) {
    const int other_1 = (direction + 1) % max_dims, other_2 = (direction + 2) % max_dims;
    const long plane = static_cast<long>(grid.storage[other_1])*grid.storage[other_2];
    const long thread = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (thread >= plane*grid.halo[direction]) return;
    const int layer = static_cast<int>(thread/plane);
    const long within = thread % plane;
    int index[max_dims];
    index[other_1] = static_cast<int>(within % grid.storage[other_1]) - grid.halo[other_1];
    index[other_2] = static_cast<int>(within/grid.storage[other_1]) - grid.halo[other_2];
    const int cells = grid.cells[direction];
    int source[max_dims] = {index[0], index[1], index[2]};
    index[direction] = -1 - layer;
    source[direction] = cells - 1 - layer;
    field[grid.storageIndex(index[0], index[1], index[2])] = field[grid.storageIndex(source[0], source[1], source[2])];
    index[direction] = cells + layer;
    source[direction] = layer;
    field[grid.storageIndex(index[0], index[1], index[2])] = field[grid.storageIndex(source[0], source[1], source[2])];
}

// tiled storage: every stored halo cell on one side of an axis, over the whole stored extent of the other axes, takes its mirror
// or, on a periodic axis, the cell one period away
__global__ void fillTiledSideKernel(GridView grid, real* field, int direction, int side, real coefficient, real value, const int* solid_markers,
                                    bool wrap) {
    const long cell = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    int index[max_dims];
    if (cell >= grid.numStorageThreads() || !grid.storageCell(cell, index[0], index[1], index[2])) return;
    const int cells = grid.cells[direction];
    if (side == 0 ? index[direction] >= 0 : index[direction] < cells) return;
    int source[max_dims] = {index[0], index[1], index[2]};
    if (wrap) {
        source[direction] += side == 0 ? cells : -cells;
        field[cell] = field[grid.storageIndex(source[0], source[1], source[2])];
        return;
    }
    source[direction] = side == 0 ? -1 - index[direction] : 2*cells - 1 - index[direction];  // mirror across the boundary face
    const long interior_cell = grid.storageIndex(source[0], source[1], source[2]);
    const real applied_value = solid_markers != nullptr && solid_markers[interior_cell] != 0 ? 0.0 : value;
    field[cell] = coefficient*field[interior_cell] + applied_value;
}

void fillTiledHalos(const GridView& grid, const HaloRule& rule, real* field) {
    for (int d = 0; d < numDims(grid); ++d)
        for (int side = 0; side < 2; ++side)
            fillTiledSideKernel<<<numBlocks(grid.numStorageThreads()), block_size>>>(grid, field, d, side, rule.coefficient[d][side],
                                                                                    rule.value[d][side], rule.solid_markers, grid.periodic[d]);
    CUDA_CHECK(cudaGetLastError());
}

void fillPhysicalSides(const GridView& grid, const HaloRule& rule, real* field, int direction) {
    const int other_1 = (direction + 1) % max_dims, other_2 = (direction + 2) % max_dims;
    const long num_threads = static_cast<long>(grid.storage[other_1])*grid.storage[other_2]*grid.halo[direction];
    if (num_threads == 0) return;
    if (grid.periodic[direction] && direction != grid.decomposed_axis) {
        wrapSidesKernel<<<numBlocks(num_threads), block_size>>>(grid, field, direction);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    for (int side = 0; side < 2; ++side) {
        const bool is_physical = side == 0 ? grid.lower_boundary[direction] : grid.upper_boundary[direction];
        if (!is_physical) continue;
        fillSideKernel<<<numBlocks(num_threads), block_size>>>(grid, field, direction, side, rule.coefficient[direction][side],
                                                              rule.value[direction][side], rule.solid_markers);
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

HaloRule makeHaloRule(const Parameters& parameters, FieldKind kind) {
    HaloRule rule;
    for (int d = 0; d < max_dims; ++d) {
        for (int side = 0; side < 2; ++side) {
            const int boundary = parameters.boundary[d][side];
            const bool is_velocity = kind != field_pressure;
            const bool is_normal = kind == d;
            if (boundary == bc_periodic) {
                setSide(rule, d, side, neumann, 0.0);  // unused, periodic halos are copied
            } else if (boundary == bc_dirichlet) {
                is_velocity ? setSide(rule, d, side, dirichlet, parameters.inflow_velocity[d][kind]) : setSide(rule, d, side, neumann, 0.0);
            } else if (boundary == bc_ghost_extrapolation) {
                is_velocity ? setSide(rule, d, side, neumann, 0.0) : setSide(rule, d, side, dirichlet, parameters.boundary_pressure[d][side]);
            } else if (boundary == bc_slip_wall) {
                setSide(rule, d, side, is_velocity && is_normal ? dirichlet : neumann, 0.0);
            } else if (boundary == bc_no_slip_wall) {
                setSide(rule, d, side, is_velocity ? dirichlet : neumann, 0.0);
            } else {
                CTM_REQUIRE(d >= parameters.num_dims, "unsupported boundary condition %d", boundary);
                setSide(rule, d, side, neumann, 0.0);
            }
        }
    }
    return rule;
}

void fillHalos(const GridView& grid, const HaloRule& rule, HaloExchange& exchange, real* field) {
    if (grid.tiled) {
        fillTiledHalos(grid, rule, field);  // one rank holds the whole domain
        return;
    }
    const int axis = grid.decomposed_axis;
    for (int d = 0; d < numDims(grid); ++d)
        if (d != axis) fillPhysicalSides(grid, rule, field, d);
    exchange.exchange(field);
    fillPhysicalSides(grid, rule, field, axis);
}
