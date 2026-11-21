#include "right_hand_side.hpp"

#include <cmath>

namespace {

constexpr int max_initial_patches = 10;

struct InitialPatches {
    int count;
    InitialPatch patches[max_initial_patches];
};

// later patches overwrite earlier ones, as in MFC
__global__ void initialConditionKernel(GridView grid, InitialPatches initial, real* velocity_x, real* velocity_y, real* velocity_z,
                                       real* pressure) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= grid.numOwned()) return;
    int index[max_dims];
    grid.ownedCell(owned, index[0], index[1], index[2]);
    const long cell = grid.storageIndex(index[0], index[1], index[2]);
    real* velocity[max_dims] = {velocity_x, velocity_y, velocity_z};
    for (int p = 0; p < initial.count; ++p) {
        const InitialPatch& patch = initial.patches[p];

        // shielded Gaussian vortex on the patch velocity, pressure from the cyclostrophic balance dp/dr = u_theta^2/r
        if (patch.geometry == patch_hardcoded_2d && patch.hcid == hcid_gaussian_vortex) {
            const real x = grid.center(0, index[0]) - patch.centroid[0], y = grid.center(1, index[1]) - patch.centroid[1];
            const real radius_squared = (x*x + y*y)/(patch.radius*patch.radius);
            const real swirl = patch.vortex_velocity/patch.radius*exp(0.5*(1.0 - radius_squared));
            velocity[0][cell] = patch.velocity[0] - swirl*y;
            velocity[1][cell] = patch.velocity[1] + swirl*x;
            velocity[2][cell] = 0.0;
            pressure[cell] = patch.pressure - 0.5*patch.vortex_velocity*patch.vortex_velocity*exp(1.0 - radius_squared);
            continue;
        }
        bool inside = true;
        for (int d = 0; d < numDims(grid); ++d) inside &= fabs(grid.center(d, index[d]) - patch.centroid[d]) <= 0.5*patch.length[d];
        if (!inside) continue;
        for (int d = 0; d < max_dims; ++d) velocity[d][cell] = d < numDims(grid) ? patch.velocity[d] : 0.0;
        pressure[cell] = patch.pressure;
    }
}

__global__ void scatterOwnedKernel(GridView grid, const real* compact, real* storage) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= grid.numOwned()) return;
    int i, j, k;
    grid.ownedCell(owned, i, j, k);
    storage[grid.storageIndex(i, j, k)] = compact[owned];
}

__global__ void gatherOwnedKernel(GridView grid, const real* storage, real* compact) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= grid.numOwned()) return;
    int i, j, k;
    grid.ownedCell(owned, i, j, k);
    compact[owned] = storage[grid.storageIndex(i, j, k)];
}

__device__ inline real laplacian(const GridView& grid, const real* field, long cell) {
    real value = 0.0;
    for (int d = 0; d < numDims(grid); ++d) {
        const long step = grid.stride(d);
        value += (field[cell - step] - 2.0*field[cell] + field[cell + step])/(grid.spacing[d]*grid.spacing[d]);
    }
    return value;
}

__device__ inline real centralGradient(const GridView& grid, const real* field, long cell, int direction) {
    const long step = grid.stride(direction);
    return (field[cell + step] - field[cell - step])/(2.0*grid.spacing[direction]);
}

// y = shift*x + scale*laplacian(x) on owned cells, x has its halos filled
__global__ void helmholtzKernel(GridView grid, const real* x_storage, real shift, real scale, real* y_owned) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= grid.numOwned()) return;
    int i, j, k;
    grid.ownedCell(owned, i, j, k);
    const long cell = grid.storageIndex(i, j, k);
    y_owned[owned] = shift*x_storage[cell] + scale*laplacian(grid, x_storage, cell);
}

// conservative convection div(U q) with divergence free face velocities and central face values
__global__ void convectionKernel(GridView grid, const real* field, const real* faces_x, const real* faces_y, const real* faces_z,
                                 real* convection) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= grid.numOwned()) return;
    int i, j, k;
    grid.ownedCell(owned, i, j, k);
    const long cell = grid.storageIndex(i, j, k);
    const real* faces[max_dims] = {faces_x, faces_y, faces_z};
    real value = 0.0;
    for (int d = 0; d < numDims(grid); ++d) {
        const long step = grid.stride(d);
        const real upper_flux = faces[d][cell + step]*0.5*(field[cell] + field[cell + step]);
        const real lower_flux = faces[d][cell]*0.5*(field[cell - step] + field[cell]);
        value += (upper_flux - lower_flux)/grid.spacing[d];
    }
    convection[owned] = value;
}

struct PredictorTerms {
    real dt, gamma, zeta, alpha, kinematic_viscosity;
};

// u - dt*(gamma*N + zeta*N_previous) - alpha*dt*grad(p) + alpha*dt*nu*(L u)/2 + alpha*dt*nu*(boundary offset), zero on solid rows
__global__ void predictorRhsKernel(GridView grid, PredictorTerms terms, int component, const real* field, const real* convection,
                                   const real* previous_convection, const real* pressure, const real* boundary_offset,
                                   const signed char* cell_types, real* rhs) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= grid.numOwned()) return;
    if (cell_types[owned] != cell_fluid) {
        rhs[owned] = 0.0;  // stationary no-slip wall
        return;
    }
    int i, j, k;
    grid.ownedCell(owned, i, j, k);
    const long cell = grid.storageIndex(i, j, k);
    const real stage_dt = terms.alpha*terms.dt;
    const real explicit_convection = terms.gamma*convection[owned] + (terms.zeta != 0.0 ? terms.zeta*previous_convection[owned] : 0.0);
    rhs[owned] = field[cell] - terms.dt*explicit_convection - stage_dt*centralGradient(grid, pressure, cell, component)
                 + 0.5*stage_dt*terms.kinematic_viscosity*laplacian(grid, field, cell)
                 + stage_dt*terms.kinematic_viscosity*boundary_offset[owned];
}

// face index f along a direction for owned cells in the other directions, faces 0..cells inclusive
__device__ inline bool faceCell(const GridView& grid, int direction, long thread, int index[max_dims]) {
    int extent[max_dims] = {grid.cells[0], grid.cells[1], grid.cells[2]};
    extent[direction] += 1;
    if (thread >= static_cast<long>(extent[0])*extent[1]*extent[2]) return false;
    index[0] = static_cast<int>(thread % extent[0]);
    index[1] = static_cast<int>((thread/extent[0]) % extent[1]);
    index[2] = static_cast<int>(thread/(static_cast<long>(extent[0])*extent[1]));
    return true;
}

inline long numFaceThreads(const GridView& grid, int direction) {
    long count = 1;
    for (int d = 0; d < max_dims; ++d) count *= grid.cells[d] + (d == direction ? 1 : 0);
    return count;
}

__global__ void interpolateFacesKernel(GridView grid, int direction, const real* field, real* faces) {
    int index[max_dims];
    if (!faceCell(grid, direction, blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x, index)) return;
    const long upper = grid.storageIndex(index[0], index[1], index[2]);
    faces[upper] = 0.5*(field[upper - grid.stride(direction)] + field[upper]);
}

// interpolated tentative velocity with the cell pressure gradient swapped for the compact face gradient on interior fluid faces
__global__ void predictedFacesKernel(GridView grid, int direction, const real* predicted, const real* pressure, const int* markers,
                                     real stage_dt, real* faces) {
    int index[max_dims];
    if (!faceCell(grid, direction, blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x, index)) return;
    const long step = grid.stride(direction);
    const long upper = grid.storageIndex(index[0], index[1], index[2]);
    const long lower = upper - step;
    real value = 0.5*(predicted[lower] + predicted[upper]);
    const int global_face = grid.offset[direction] + index[direction];
    const bool is_interior = grid.periodic[direction] || (global_face > 0 && global_face < grid.global_cells[direction]);
    const bool is_interior_fluid = is_interior && markers[lower] == 0 && markers[upper] == 0;
    if (is_interior_fluid) {
        const real cell_gradient = 0.5*(centralGradient(grid, pressure, lower, direction) + centralGradient(grid, pressure, upper, direction));
        value += stage_dt*(cell_gradient - (pressure[upper] - pressure[lower])/grid.spacing[direction]);
    }
    faces[upper] = value;
}

__global__ void divergenceKernel(GridView grid, const real* faces_x, const real* faces_y, const real* faces_z, const signed char* cell_types,
                                 real scale, bool fluid_only, real* divergence) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= grid.numOwned()) return;
    if (fluid_only && cell_types[owned] != cell_fluid) {
        divergence[owned] = 0.0;
        return;
    }
    int i, j, k;
    grid.ownedCell(owned, i, j, k);
    const long cell = grid.storageIndex(i, j, k);
    const real* faces[max_dims] = {faces_x, faces_y, faces_z};
    real value = 0.0;
    for (int d = 0; d < numDims(grid); ++d) value += (faces[d][cell + grid.stride(d)] - faces[d][cell])/grid.spacing[d];
    divergence[owned] = scale*value;
}

__global__ void correctFacesKernel(GridView grid, int direction, const real* correction, real stage_dt, real* faces) {
    int index[max_dims];
    if (!faceCell(grid, direction, blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x, index)) return;
    const long upper = grid.storageIndex(index[0], index[1], index[2]);
    faces[upper] -= stage_dt*(correction[upper] - correction[upper - grid.stride(direction)])/grid.spacing[direction];
}

__global__ void correctVelocityKernel(GridView grid, int component, const real* predicted, const real* correction, real stage_dt,
                                      real* velocity) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= grid.numOwned()) return;
    int i, j, k;
    grid.ownedCell(owned, i, j, k);
    const long cell = grid.storageIndex(i, j, k);
    velocity[cell] = predicted[cell] - stage_dt*centralGradient(grid, correction, cell, component);
}

__global__ void addOwnedKernel(GridView grid, const real* compact, real* storage) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= grid.numOwned()) return;
    int i, j, k;
    grid.ownedCell(owned, i, j, k);
    storage[grid.storageIndex(i, j, k)] += compact[owned];
}

// largest |value| over the owned cells selected by cell_types, a NaN wins because its bits exceed every finite double
__global__ void maxAbsoluteKernel(long count, const real* values, const signed char* cell_types, unsigned long long* maximum) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= count || (cell_types != nullptr && cell_types[owned] != cell_fluid)) return;
    atomicMax(maximum, static_cast<unsigned long long>(__double_as_longlong(fabs(values[owned]))));
}

real maxAbsolute(long count, const real* values, const signed char* cell_types, DeviceArray<real>& reduction) {
    reduction.fillZero();
    maxAbsoluteKernel<<<numBlocks(count), block_size>>>(count, values, cell_types, reinterpret_cast<unsigned long long*>(reduction.data()));
    CUDA_CHECK(cudaGetLastError());
    real local;
    reduction.copyToHost(&local);
    return allreduceMax(local);
}

}  // namespace

std::vector<StageCoefficients> rungeKuttaStages(int rk_order) {
    if (rk_order == 3) return {std::begin(rk3_stages), std::end(rk3_stages)};
    if (rk_order == 2) return {std::begin(rk2_stages), std::end(rk2_stages)};
    return {std::begin(rk1_stages), std::end(rk1_stages)};
}

FlowSolver::FlowSolver(const Parameters& parameters, const GridView& grid, ImmersedBoundaries& immersed_boundaries)
    : parameters(parameters), grid(grid), immersed_boundaries(immersed_boundaries),
      stages(rungeKuttaStages(parameters.rk_order)), num_dims(grid.num_dims), exchange(grid), transpose(grid),
      gmres(grid.numOwned(), parameters.krylov_restart, parameters.krylov_tolerance, parameters.krylov_max_iterations) {
    const long num_storage = grid.numStorage(), num_owned = grid.numOwned();
    for (int d = 0; d < max_dims; ++d) {
        this->velocity[d].resize(num_storage);
        this->faces[d].resize(num_storage);
        this->predicted[d].resize(num_storage);
        this->convection[d].resize(num_owned);
        this->previous_convection[d].resize(num_owned);
        this->boundary_offsets[d].resize(num_owned);
    }
    this->pressure.resize(num_storage);
    this->rhs.resize(num_owned);
    this->solution.resize(num_owned);
    this->pressure_correction.resize(num_owned);
    this->operator_input.resize(num_storage);
    this->correction_field.resize(num_storage);
    this->reduction.resize(1);

    for (int kind = 0; kind < num_field_kinds; ++kind) {
        this->rules[kind] = makeHaloRule(parameters, static_cast<FieldKind>(kind));
        this->homogeneous_rules[kind] = this->rules[kind].homogeneous();
    }

    // one Helmholtz preconditioner per stage and component, one Poisson preconditioner
    for (const StageCoefficients& coefficients : this->stages)
        for (int component = 0; component < this->num_dims; ++component) {
            const real diffusion_number = 0.5*coefficients.alpha*parameters.dt*parameters.kinematic_viscosity;
            this->helmholtz_preconditioners.push_back(std::make_unique<SpectralSolver>(
                grid, this->transpose, this->transforms, this->homogeneous_rules[component], 1.0, -diffusion_number));
        }
    this->poisson_preconditioner = std::make_unique<SpectralSolver>(grid, this->transpose, this->transforms,
                                                                    this->homogeneous_rules[field_pressure], 0.0, 1.0);

    // laplacian of the boundary values alone, the constant part of the velocity operators
    for (int component = 0; component < this->num_dims; ++component) {
        this->operator_input.fillZero();
        fillHalos(grid, this->rules[component], this->exchange, this->operator_input.data());
        helmholtzKernel<<<numBlocks(num_owned), block_size>>>(grid, this->operator_input.data(), 0.0, 1.0, this->boundary_offsets[component].data());
    }
    CUDA_CHECK(cudaGetLastError());
}

void FlowSolver::fillVelocityHalos(DeviceArray<real>* fields) {
    for (int component = 0; component < this->num_dims; ++component)
        fillHalos(this->grid, this->rules[component], this->exchange, fields[component].data());
}

void FlowSolver::applyOperator(const real* x, real* y, FieldKind kind, real shift, real scale, real ghost_sign) {
    const GridView& g = this->grid;
    scatterOwnedKernel<<<numBlocks(g.numOwned()), block_size>>>(g, x, this->operator_input.data());
    fillHalos(g, this->homogeneous_rules[kind], this->exchange, this->operator_input.data());
    helmholtzKernel<<<numBlocks(g.numOwned()), block_size>>>(g, this->operator_input.data(), shift, scale, y);
    CUDA_CHECK(cudaGetLastError());
    this->immersed_boundaries.applyGhostRows(this->operator_input.data(), y, shift, scale, this->homogeneous_rules[kind], ghost_sign);
}

void FlowSolver::initialize() {
    const GridView& g = this->grid;
    InitialPatches initial;
    initial.count = static_cast<int>(this->parameters.initial_patches.size());
    CTM_REQUIRE(initial.count <= max_initial_patches, "at most %d initial condition patches", max_initial_patches);
    for (int p = 0; p < initial.count; ++p) initial.patches[p] = this->parameters.initial_patches[p];
    initialConditionKernel<<<numBlocks(g.numOwned()), block_size>>>(g, initial, this->velocity[0].data(), this->velocity[1].data(),
                                                                    this->velocity[2].data(), this->pressure.data());
    CUDA_CHECK(cudaGetLastError());

    // immersed boundaries on the initial field, then project it onto a divergence free one without keeping the pressure
    this->fillVelocityHalos(this->velocity);
    for (int component = 0; component < this->num_dims; ++component) this->immersed_boundaries.applyGhostVelocity(this->velocity[component].data());
    this->fillVelocityHalos(this->velocity);
    fillHalos(g, this->rules[field_pressure], this->exchange, this->pressure.data());
    for (int d = 0; d < this->num_dims; ++d) {
        interpolateFacesKernel<<<numBlocks(numFaceThreads(g, d)), block_size>>>(g, d, this->velocity[d].data(), this->faces[d].data());
        CUDA_CHECK(cudaMemcpy(this->predicted[d].data(), this->velocity[d].data(), g.numStorage()*sizeof(real), cudaMemcpyDeviceToDevice));
    }
    DeviceArray<real> initial_pressure(g.numStorage());
    CUDA_CHECK(cudaMemcpy(initial_pressure.data(), this->pressure.data(), g.numStorage()*sizeof(real), cudaMemcpyDeviceToDevice));
    this->project(this->parameters.dt);
    CUDA_CHECK(cudaMemcpy(this->pressure.data(), initial_pressure.data(), g.numStorage()*sizeof(real), cudaMemcpyDeviceToDevice));
}

void FlowSolver::advance() {
    for (int stage_index = 0; stage_index < static_cast<int>(this->stages.size()); ++stage_index) this->stage(stage_index);
}

void FlowSolver::stage(int stage_index) {
    const GridView& g = this->grid;
    const StageCoefficients& coefficients = this->stages[stage_index];
    const real stage_dt = coefficients.alpha*this->parameters.dt;
    const real diffusion_number = 0.5*stage_dt*this->parameters.kinematic_viscosity;
    const long num_owned = g.numOwned();
    const signed char* cell_types = this->immersed_boundaries.geometry.cell_types.data();

    // explicit convection with the current divergence free face velocities
    for (int component = 0; component < this->num_dims; ++component)
        convectionKernel<<<numBlocks(num_owned), block_size>>>(g, this->velocity[component].data(), this->faces[0].data(), this->faces[1].data(),
                                                               this->faces[2].data(), this->convection[component].data());

    // get tentative velocity with the old pressure gradient, Crank-Nicolson diffusion solved by GMRES
    const PredictorTerms terms = {this->parameters.dt, coefficients.gamma, coefficients.zeta, coefficients.alpha, this->parameters.kinematic_viscosity};
    for (int component = 0; component < this->num_dims; ++component) {
        predictorRhsKernel<<<numBlocks(num_owned), block_size>>>(g, terms, component, this->velocity[component].data(),
                                                                 this->convection[component].data(), this->previous_convection[component].data(),
                                                                 this->pressure.data(), this->boundary_offsets[component].data(), cell_types,
                                                                 this->rhs.data());
        gatherOwnedKernel<<<numBlocks(num_owned), block_size>>>(g, this->velocity[component].data(), this->solution.data());
        SpectralSolver& preconditioner = *this->helmholtz_preconditioners[stage_index*this->num_dims + component];
        const FieldKind kind = static_cast<FieldKind>(component);
        this->iterations[component] = this->gmres.solve(
            [&](const real* x, real* y) { this->applyOperator(x, y, kind, 1.0, -diffusion_number, 1.0); },
            [&](const real* x, real* y) { preconditioner.solve(x, y); }, this->rhs.data(), this->solution.data());
        scatterOwnedKernel<<<numBlocks(num_owned), block_size>>>(g, this->solution.data(), this->predicted[component].data());
    }
    this->fillVelocityHalos(this->predicted);

    // face velocities with the cell pressure gradient swapped for the compact face gradient
    for (int d = 0; d < this->num_dims; ++d)
        predictedFacesKernel<<<numBlocks(numFaceThreads(g, d)), block_size>>>(g, d, this->predicted[d].data(), this->pressure.data(),
                                                                              this->immersed_boundaries.geometry.markers.data(), stage_dt,
                                                                              this->faces[d].data());
    CUDA_CHECK(cudaGetLastError());
    // with a single projection the earlier stages carry the tentative velocity forward, the last one removes the
    // divergence of the whole step, so its pressure increment spans the full dt
    const bool is_last_stage = stage_index == static_cast<int>(this->stages.size()) - 1;
    if (this->parameters.rk_single_projection && !is_last_stage) {
        for (int component = 0; component < this->num_dims; ++component)
            CUDA_CHECK(cudaMemcpy(this->velocity[component].data(), this->predicted[component].data(), g.numStorage()*sizeof(real),
                                  cudaMemcpyDeviceToDevice));
        for (int component = 0; component < this->num_dims; ++component) this->immersed_boundaries.applyGhostVelocity(this->velocity[component].data());
        this->fillVelocityHalos(this->velocity);
    } else {
        this->project(this->parameters.rk_single_projection ? this->parameters.dt : stage_dt);
    }
    for (int component = 0; component < this->num_dims; ++component)
        std::swap(this->convection[component], this->previous_convection[component]);
}

void FlowSolver::project(real stage_dt) {
    const GridView& g = this->grid;
    const long num_owned = g.numOwned();
    const signed char* cell_types = this->immersed_boundaries.geometry.cell_types.data();

    // pressure correction from the divergence of the predicted face velocities, zero normal gradient mirrored into the ghost cells
    divergenceKernel<<<numBlocks(num_owned), block_size>>>(g, this->faces[0].data(), this->faces[1].data(), this->faces[2].data(), cell_types,
                                                           1.0/stage_dt, true, this->rhs.data());
    this->iterations[field_pressure] = this->gmres.solve(
        [&](const real* x, real* y) { this->applyOperator(x, y, field_pressure, 0.0, 1.0, -1.0); },
        [&](const real* x, real* y) { this->poisson_preconditioner->solve(x, y); }, this->rhs.data(), this->pressure_correction.data());

    // make the faces divergence free, correct the cell velocities, and accumulate the pressure
    scatterOwnedKernel<<<numBlocks(num_owned), block_size>>>(g, this->pressure_correction.data(), this->correction_field.data());
    fillHalos(g, this->homogeneous_rules[field_pressure], this->exchange, this->correction_field.data());
    for (int d = 0; d < this->num_dims; ++d) {
        correctFacesKernel<<<numBlocks(numFaceThreads(g, d)), block_size>>>(g, d, this->correction_field.data(), stage_dt, this->faces[d].data());
        correctVelocityKernel<<<numBlocks(num_owned), block_size>>>(g, d, this->predicted[d].data(), this->correction_field.data(), stage_dt,
                                                                    this->velocity[d].data());
    }
    addOwnedKernel<<<numBlocks(num_owned), block_size>>>(g, this->pressure_correction.data(), this->pressure.data());
    CUDA_CHECK(cudaGetLastError());

    // ghost cells follow the corrected fluid, then every halo is refreshed for the next stage and the forces
    this->fillVelocityHalos(this->velocity);
    for (int component = 0; component < this->num_dims; ++component) this->immersed_boundaries.applyGhostVelocity(this->velocity[component].data());
    this->fillVelocityHalos(this->velocity);
    fillHalos(g, this->rules[field_pressure], this->exchange, this->pressure.data());
}

real FlowSolver::maxDivergence() {
    divergenceKernel<<<numBlocks(this->grid.numOwned()), block_size>>>(this->grid, this->faces[0].data(), this->faces[1].data(),
                                                                       this->faces[2].data(), this->immersed_boundaries.geometry.cell_types.data(),
                                                                       1.0, true, this->rhs.data());
    return maxAbsolute(this->grid.numOwned(), this->rhs.data(), nullptr, this->reduction);
}

bool FlowSolver::isFinite() {
    for (int component = 0; component < this->num_dims; ++component) {
        gatherOwnedKernel<<<numBlocks(this->grid.numOwned()), block_size>>>(this->grid, this->velocity[component].data(), this->rhs.data());
        if (!std::isfinite(maxAbsolute(this->grid.numOwned(), this->rhs.data(), nullptr, this->reduction))) return false;
    }
    return true;
}

std::vector<real> FlowSolver::forces() {
    real* velocity_pointers[max_dims] = {this->velocity[0].data(), this->velocity[1].data(), this->velocity[2].data()};
    return this->immersed_boundaries.computeForces(this->pressure.data(), velocity_pointers, this->parameters.kinematic_viscosity);
}
