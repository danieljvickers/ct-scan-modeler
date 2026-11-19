#include "boundary_conditions.hpp"

namespace {

constexpr real neumann = 1.0;     // halo copies the interior, zero gradient
constexpr real dirichlet = -1.0;  // halo mirrors with opposite sign, zero value at the face

void setSide(HaloRule& rule, int direction, int side, real coefficient, real face_value) {
    rule.coefficient[direction][side] = coefficient;
    rule.value[direction][side] = coefficient == dirichlet ? 2.0*face_value : 0.0;  // linear through the face value
}

// fill the halo layers of one side along one axis, over the whole storage extent of the other axes
__global__ void fillSideKernel(GridView grid, real* field, int direction, int side, real coefficient, real value) {
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

    field[grid.storageIndex(halo_index[0], halo_index[1], halo_index[2])] =
        coefficient*field[grid.storageIndex(interior_index[0], interior_index[1], interior_index[2])] + value;
}

void fillPhysicalSides(const GridView& grid, const HaloRule& rule, real* field, int direction) {
    const int other_1 = (direction + 1) % max_dims, other_2 = (direction + 2) % max_dims;
    const long num_threads = static_cast<long>(grid.storage[other_1])*grid.storage[other_2]*grid.halo[direction];
    if (num_threads == 0) return;
    for (int side = 0; side < 2; ++side) {
        const bool is_physical = side == 0 ? grid.lower_boundary[direction] : grid.upper_boundary[direction];
        if (!is_physical) continue;
        fillSideKernel<<<numBlocks(num_threads), block_size>>>(grid, field, direction, side, rule.coefficient[direction][side],
                                                              rule.value[direction][side]);
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
            if (boundary == bc_dirichlet) {
                is_velocity ? setSide(rule, d, side, dirichlet, parameters.inflow_velocity[d][kind]) : setSide(rule, d, side, neumann, 0.0);
            } else if (boundary == bc_ghost_extrapolation) {
                is_velocity ? setSide(rule, d, side, neumann, 0.0) : setSide(rule, d, side, dirichlet, 0.0);
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
    const int axis = grid.decomposed_axis;
    for (int d = 0; d < numDims(grid); ++d)
        if (d != axis) fillPhysicalSides(grid, rule, field, d);
    exchange.exchange(field);
    fillPhysicalSides(grid, rule, field, axis);
}
