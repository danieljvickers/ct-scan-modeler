#include "ibm.hpp"

#include "mpi_proxy.hpp"

namespace {

__global__ void ghostVelocityKernel(int num_ghost_points, const long* storage_cells, const long* stencil_cells, const real* weights,
                                    real* velocity) {
    const int ghost = blockIdx.x*blockDim.x + threadIdx.x;
    if (ghost >= num_ghost_points) return;
    real image_value = 0.0;
    for (int corner = 0; corner < max_stencil; ++corner)
        image_value += weights[ghost*max_stencil + corner]*velocity[stencil_cells[ghost*max_stencil + corner]];
    velocity[storage_cells[ghost]] = -image_value;
}

__global__ void zeroInteriorKernel(GridView grid, const signed char* cell_types, real* velocity) {
    int i, j, k;
    long entry, cell;
    if (!grid.ownedThread(blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x, i, j, k, entry, cell) || cell_types[entry] != cell_interior) return;
    velocity[cell] = 0.0;
}

__global__ void ghostRowsKernel(GridView grid, HaloRule rule, int num_ghost_points, const long* storage_cells, const long* owned_cells,
                                const long* stencil_cells, const real* weights, const real* x_storage, real* y_owned, real shift,
                                real scale, real ghost_sign) {
    const int ghost = blockIdx.x*blockDim.x + threadIdx.x;
    if (ghost >= num_ghost_points) return;

    // diagonal of shift + scale*laplacian, the halo coefficient replaces the missing neighbor at a domain boundary
    int index[max_dims];
    grid.entryCell(owned_cells[ghost], index[0], index[1], index[2]);
    real laplacian_diagonal = 0.0;
    for (int d = 0; d < numDims(grid); ++d) {
        const int global_index = index[d] + grid.offset[d];
        real diagonal = -2.0;
        if (!grid.periodic[d] && global_index == 0) diagonal += rule.coefficient[d][0];
        if (!grid.periodic[d] && global_index == grid.global_cells[d] - 1) diagonal += rule.coefficient[d][1];
        laplacian_diagonal += diagonal/(grid.spacing[d]*grid.spacing[d]);
    }

    real image_value = 0.0;
    for (int corner = 0; corner < max_stencil; ++corner)
        image_value += weights[ghost*max_stencil + corner]*x_storage[stencil_cells[ghost*max_stencil + corner]];
    y_owned[owned_cells[ghost]] = (shift + scale*laplacian_diagonal)*(x_storage[storage_cells[ghost]] + ghost_sign*image_value);
}

// -grad(p) + nu*laplacian(u) times the cell volume, summed per patch over owned solid cells
__global__ void forcesKernel(GridView grid, const int* markers, const real* pressure, const real* velocity_x, const real* velocity_y,
                             const real* velocity_z, real kinematic_viscosity, real* forces) {
    int i, j, k;
    long entry, cell;
    if (!grid.ownedThread(blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x, i, j, k, entry, cell)) return;
    const int patch = markers[cell];
    if (patch == 0) return;
    const real* velocity[max_dims] = {velocity_x, velocity_y, velocity_z};
    for (int component = 0; component < numDims(grid); ++component) {
        real force = -(pressure[grid.shifted(cell, i, j, k, component, 1)] - pressure[grid.shifted(cell, i, j, k, component, -1)])
                     /(2.0*grid.spacing[component]);
        for (int d = 0; d < numDims(grid); ++d) {
            const real* field = velocity[component];
            force += kinematic_viscosity*(field[grid.shifted(cell, i, j, k, d, -1)] - 2.0*field[cell] + field[grid.shifted(cell, i, j, k, d, 1)])
                     /(grid.spacing[d]*grid.spacing[d]);
        }
        atomicAdd(&forces[(patch - 1)*max_dims + component], force*grid.cellVolume());  // many cells add to one patch
    }
}

}  // namespace

ImmersedBoundaries::ImmersedBoundaries(const Parameters& parameters, const GridView& grid)
    : grid(grid), geometry(parameters, grid), is_active(parameters.ib) {}

void ImmersedBoundaries::applyGhostVelocity(real* velocity) const {
    if (!this->is_active) return;
    const GhostPoints& ghosts = this->geometry.ghost_points;
    if (ghosts.count > 0)
        ghostVelocityKernel<<<numBlocks(ghosts.count), block_size>>>(ghosts.count, ghosts.storage_cells.data(), ghosts.stencil_cells.data(),
                                                                      ghosts.weights.data(), velocity);
    zeroInteriorKernel<<<numBlocks(this->grid.numOwnedThreads()), block_size>>>(this->grid, this->geometry.cell_types.data(), velocity);
    CUDA_CHECK(cudaGetLastError());
}

void ImmersedBoundaries::applyGhostRows(const real* x_storage, real* y_owned, real shift, real scale, const HaloRule& rule,
                                        real ghost_sign) const {
    const GhostPoints& ghosts = this->geometry.ghost_points;
    if (!this->is_active || ghosts.count == 0) return;
    ghostRowsKernel<<<numBlocks(ghosts.count), block_size>>>(this->grid, rule, ghosts.count, ghosts.storage_cells.data(),
                                                              ghosts.owned_cells.data(), ghosts.stencil_cells.data(), ghosts.weights.data(),
                                                              x_storage, y_owned, shift, scale, ghost_sign);
    CUDA_CHECK(cudaGetLastError());
}

std::vector<real> ImmersedBoundaries::computeForces(const real* pressure, real* const* velocity, real kinematic_viscosity) const {
    std::vector<real> forces(static_cast<size_t>(this->geometry.num_patches)*max_dims, 0.0);
    if (!this->is_active) return forces;
    DeviceArray<real> device_forces(forces.size());
    forcesKernel<<<numBlocks(this->grid.numOwnedThreads()), block_size>>>(this->grid, this->geometry.markers.data(), pressure, velocity[0],
                                                                    velocity[1], velocity[2], kinematic_viscosity, device_forces.data());
    CUDA_CHECK(cudaGetLastError());
    device_forces.copyToHost(forces.data());
    allreduceSum(forces.data(), static_cast<int>(forces.size()));
    return forces;
}
