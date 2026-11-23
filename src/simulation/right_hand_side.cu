#include "right_hand_side.hpp"

#include <algorithm>

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

// with blocked faces a fluid cell exchanges no pressure flux with its solid neighbors, the discrete match of zero normal face velocity
__global__ void blockedLaplacianKernel(GridView grid, const real* x_storage, const int* markers, real shift, real scale, real* y_owned) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= grid.numOwned()) return;
    int i, j, k;
    grid.ownedCell(owned, i, j, k);
    const long cell = grid.storageIndex(i, j, k);
    if (markers[cell] != 0) {
        y_owned[owned] = shift*x_storage[cell] + scale*laplacian(grid, x_storage, cell);  // solid rows, ghost rows are overwritten later
        return;
    }
    real value = 0.0;
    for (int d = 0; d < numDims(grid); ++d)
        for (int side = -1; side <= 1; side += 2) {
            const long neighbor = cell + side*grid.stride(d);
            if (markers[neighbor] == 0) value += (x_storage[neighbor] - x_storage[cell])/(grid.spacing[d]*grid.spacing[d]);
        }
    y_owned[owned] = shift*x_storage[cell] + scale*value;
}

// a cell is next to the wall if it or any face neighbor is solid
__device__ inline bool nearWall(const GridView& grid, const int* markers, long cell) {
    if (markers[cell] != 0) return true;
    for (int d = 0; d < numDims(grid); ++d)
        if (markers[cell - grid.stride(d)] != 0 || markers[cell + grid.stride(d)] != 0) return true;
    return false;
}

// WENO3 face value from the upstream side, Jiang and Shu (1996); far, near, next are the cells two upstream, one upstream, and downstream
__device__ inline real weno3(real far, real near, real next) {
    constexpr real epsilon = 1.0e-6;
    const real candidate_upwind = -0.5*far + 1.5*near;  // stencil {far, near}
    const real candidate_central = 0.5*(near + next);   // stencil {near, next}
    const real smooth_upwind = (near - far)*(near - far), smooth_central = (next - near)*(next - near);
    const real alpha_upwind = (1.0/3.0)/((epsilon + smooth_upwind)*(epsilon + smooth_upwind));
    const real alpha_central = (2.0/3.0)/((epsilon + smooth_central)*(epsilon + smooth_central));
    return (alpha_upwind*candidate_upwind + alpha_central*candidate_central)/(alpha_upwind + alpha_central);
}

struct ConvectionScheme {
    const int* markers;  // null without immersed boundaries
    real wall_upwind;
    int weno_order;
    bool weno_wall_only;
};

// flux through the face between lower and upper: central by default, WENO3 or a blend toward upwind where asked
__device__ inline real faceFlux(const GridView& grid, const ConvectionScheme& scheme, real face_velocity, long lower, long upper,
                                const real* field) {
    const real central = 0.5*(field[lower] + field[upper]);
    const int* markers = scheme.markers;
    const bool near_wall = markers != nullptr && (nearWall(grid, markers, lower) || nearWall(grid, markers, upper));
    if (scheme.weno_order == 3 && (!scheme.weno_wall_only || near_wall)) {
        const long step = upper - lower;
        const real value = face_velocity >= 0.0 ? weno3(field[lower - step], field[lower], field[upper])
                                                : weno3(field[upper + step], field[upper], field[lower]);
        return face_velocity*value;
    }
    const real wall_upwind = scheme.wall_upwind;
    if (wall_upwind <= 0.0 || !near_wall) return face_velocity*central;
    const real upwind = face_velocity >= 0.0 ? field[lower] : field[upper];
    return face_velocity*((1.0 - wall_upwind)*central + wall_upwind*upwind);
}

// conservative convection div(U q) with divergence free face velocities; both cells of a face compute the same flux
__global__ void convectionKernel(GridView grid, const real* field, const real* faces_x, const real* faces_y, const real* faces_z,
                                 ConvectionScheme scheme, real* convection) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= grid.numOwned()) return;
    int i, j, k;
    grid.ownedCell(owned, i, j, k);
    const long cell = grid.storageIndex(i, j, k);
    const real* faces[max_dims] = {faces_x, faces_y, faces_z};
    real value = 0.0;
    for (int d = 0; d < numDims(grid); ++d) {
        const long step = grid.stride(d);
        const real upper_flux = faceFlux(grid, scheme, faces[d][cell + step], cell, cell + step, field);
        const real lower_flux = faceFlux(grid, scheme, faces[d][cell], cell - step, cell, field);
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

// block_markers is null unless faces between fluid and solid cells carry no flow
__device__ inline bool isBlockedFace(const int* block_markers, long lower, long upper) {
    return block_markers != nullptr && (block_markers[lower] != 0 || block_markers[upper] != 0);
}

__global__ void interpolateFacesKernel(GridView grid, int direction, const real* field, const int* block_markers, real* faces) {
    int index[max_dims];
    if (!faceCell(grid, direction, blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x, index)) return;
    const long upper = grid.storageIndex(index[0], index[1], index[2]);
    const long lower = upper - grid.stride(direction);
    faces[upper] = isBlockedFace(block_markers, lower, upper) ? 0.0 : 0.5*(field[lower] + field[upper]);
}

// interpolated tentative velocity with the cell pressure gradient swapped for the compact face gradient on interior fluid faces
__global__ void predictedFacesKernel(GridView grid, int direction, const real* predicted, const real* pressure, const int* markers,
                                     const int* block_markers, real stage_dt, real* faces) {
    int index[max_dims];
    if (!faceCell(grid, direction, blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x, index)) return;
    const long step = grid.stride(direction);
    const long upper = grid.storageIndex(index[0], index[1], index[2]);
    const long lower = upper - step;
    if (isBlockedFace(block_markers, lower, upper)) {
        faces[upper] = 0.0;
        return;
    }
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

__global__ void correctFacesKernel(GridView grid, int direction, const real* correction, const int* block_markers, real stage_dt, real* faces) {
    int index[max_dims];
    if (!faceCell(grid, direction, blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x, index)) return;
    const long upper = grid.storageIndex(index[0], index[1], index[2]);
    const long lower = upper - grid.stride(direction);
    if (isBlockedFace(block_markers, lower, upper)) return;
    faces[upper] -= stage_dt*(correction[upper] - correction[lower])/grid.spacing[direction];
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

enum DiagnosticQuantity : int { quantity_speed = 0, quantity_cfl = 1, quantity_pressure = 2 };

// one diagnostic value per owned cell, the CFL number is the sum over directions of |u_d| dt / dx_d
__global__ void diagnosticValuesKernel(GridView grid, int quantity, real dt, const real* u, const real* v, const real* w, const real* pressure,
                                       real* compact) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= grid.numOwned()) return;
    int i, j, k;
    grid.ownedCell(owned, i, j, k);
    const long cell = grid.storageIndex(i, j, k);
    const real speeds[max_dims] = {u[cell], grid.num_dims > 1 ? v[cell] : 0.0, grid.num_dims > 2 ? w[cell] : 0.0};
    if (quantity == quantity_speed) {
        compact[owned] = sqrt(speeds[0]*speeds[0] + speeds[1]*speeds[1] + speeds[2]*speeds[2]);
    } else if (quantity == quantity_cfl) {
        compact[owned] = dt*(fabs(speeds[0])/grid.spacing[0] + fabs(speeds[1])/grid.spacing[1] + fabs(speeds[2])/grid.spacing[2]);
    } else {
        compact[owned] = pressure[cell];
    }
}

// lowest owned index whose |value| equals the target
__global__ void findFirstKernel(long count, const real* values, real target, const signed char* cell_types, unsigned long long* first) {
    const long owned = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (owned >= count || (cell_types != nullptr && cell_types[owned] != cell_fluid)) return;
    if (fabs(values[owned]) == target) atomicMin(first, static_cast<unsigned long long>(owned));
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
        if (immersed_boundaries.isActive()) this->rules[kind].solid_markers = immersed_boundaries.geometry.markers.data();
        this->homogeneous_rules[kind] = this->rules[kind].homogeneous();
        this->full_rules[kind] = this->rules[kind];
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

    this->computeBoundaryOffsets();
}

// laplacian of the boundary values alone, the constant part of the velocity operators
void FlowSolver::computeBoundaryOffsets() {
    for (int component = 0; component < this->num_dims; ++component) {
        this->operator_input.fillZero();
        fillHalos(this->grid, this->rules[component], this->exchange, this->operator_input.data());
        helmholtzKernel<<<numBlocks(this->grid.numOwned()), block_size>>>(this->grid, this->operator_input.data(), 0.0, 1.0,
                                                                           this->boundary_offsets[component].data());
    }
    CUDA_CHECK(cudaGetLastError());
}

void FlowSolver::rampBoundaryValues(real time) {
    if (this->parameters.bc_ramp_time <= 0.0) return;
    const real fraction = std::min(1.0, time/this->parameters.bc_ramp_time);
    for (int kind = 0; kind < num_field_kinds; ++kind)
        for (int d = 0; d < max_dims; ++d)
            for (int side = 0; side < 2; ++side) this->rules[kind].value[d][side] = fraction*this->full_rules[kind].value[d][side];
    this->computeBoundaryOffsets();
}

void FlowSolver::fillVelocityHalos(DeviceArray<real>* fields) {
    for (int component = 0; component < this->num_dims; ++component)
        fillHalos(this->grid, this->rules[component], this->exchange, fields[component].data());
}

void FlowSolver::applyOperator(const real* x, real* y, FieldKind kind, real shift, real scale, real ghost_sign) {
    const GridView& g = this->grid;
    scatterOwnedKernel<<<numBlocks(g.numOwned()), block_size>>>(g, x, this->operator_input.data());
    fillHalos(g, this->homogeneous_rules[kind], this->exchange, this->operator_input.data());
    if (kind == field_pressure && this->blockMarkers() != nullptr)
        blockedLaplacianKernel<<<numBlocks(g.numOwned()), block_size>>>(g, this->operator_input.data(), this->blockMarkers(), shift, scale, y);
    else
        helmholtzKernel<<<numBlocks(g.numOwned()), block_size>>>(g, this->operator_input.data(), shift, scale, y);
    CUDA_CHECK(cudaGetLastError());
    this->immersed_boundaries.applyGhostRows(this->operator_input.data(), y, shift, scale, this->homogeneous_rules[kind], ghost_sign);
}

void FlowSolver::initialize() {
    this->time = this->parameters.t_step_start*this->parameters.dt;
    this->rampBoundaryValues(this->time);
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
        interpolateFacesKernel<<<numBlocks(numFaceThreads(g, d)), block_size>>>(g, d, this->velocity[d].data(), this->blockMarkers(), this->faces[d].data());
        CUDA_CHECK(cudaMemcpy(this->predicted[d].data(), this->velocity[d].data(), g.numStorage()*sizeof(real), cudaMemcpyDeviceToDevice));
    }
    DeviceArray<real> initial_pressure(g.numStorage());
    CUDA_CHECK(cudaMemcpy(initial_pressure.data(), this->pressure.data(), g.numStorage()*sizeof(real), cudaMemcpyDeviceToDevice));
    this->project(this->parameters.dt);
    CUDA_CHECK(cudaMemcpy(this->pressure.data(), initial_pressure.data(), g.numStorage()*sizeof(real), cudaMemcpyDeviceToDevice));
}

void FlowSolver::advance() {
    // boundary values at the end of the step while the ramp is on
    if (this->time < this->parameters.bc_ramp_time) this->rampBoundaryValues(this->time + this->parameters.dt);
    for (int stage_index = 0; stage_index < static_cast<int>(this->stages.size()); ++stage_index) this->stage(stage_index);
    this->time += this->parameters.dt;
}

void FlowSolver::stage(int stage_index) {
    const GridView& g = this->grid;
    const StageCoefficients& coefficients = this->stages[stage_index];
    const real stage_dt = coefficients.alpha*this->parameters.dt;
    const real diffusion_number = 0.5*stage_dt*this->parameters.kinematic_viscosity;
    const long num_owned = g.numOwned();
    const signed char* cell_types = this->immersed_boundaries.geometry.cell_types.data();

    // explicit convection with the current divergence free face velocities
    const ConvectionScheme scheme = {this->immersed_boundaries.isActive() ? this->immersed_boundaries.geometry.markers.data() : nullptr,
                                     this->parameters.ib_wall_upwind, this->parameters.weno_order, this->parameters.weno_wall_only};
    for (int component = 0; component < this->num_dims; ++component)
        convectionKernel<<<numBlocks(num_owned), block_size>>>(g, this->velocity[component].data(), this->faces[0].data(), this->faces[1].data(),
                                                               this->faces[2].data(), scheme, this->convection[component].data());

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
                                                                              this->immersed_boundaries.geometry.markers.data(), this->blockMarkers(), stage_dt,
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
        correctFacesKernel<<<numBlocks(numFaceThreads(g, d)), block_size>>>(g, d, this->correction_field.data(), this->blockMarkers(), stage_dt,
                                                                           this->faces[d].data());
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

FieldPeak FlowSolver::locatePeak(const signed char* cell_types) {
    FieldPeak peak;
    const long count = this->grid.numOwned();
    peak.value = maxAbsolute(count, this->rhs.data(), cell_types, this->reduction);
    if (!std::isfinite(peak.value)) return peak;

    DeviceArray<unsigned long long> first(1);
    const unsigned long long none = ~0ULL;
    first.copyFromHost(&none, 1);
    findFirstKernel<<<numBlocks(count), block_size>>>(count, this->rhs.data(), peak.value, cell_types, first.data());
    CUDA_CHECK(cudaGetLastError());
    unsigned long long local;
    first.copyToHost(&local);

    // only the rank holding the cell contributes, the sum then carries its description to every rank
    real packed[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // found, global i j k, cell type + 1, solid neighbors + 1, on face
    if (local != none) {
        int i, j, k;
        this->grid.ownedCell(static_cast<long>(local), i, j, k);
        const Geometry& geometry = this->immersed_boundaries.geometry;
        signed char type = -1;
        int solid_neighbors = -1;
        if (this->immersed_boundaries.isActive()) {
            CUDA_CHECK(cudaMemcpy(&type, geometry.cell_types.data() + local, sizeof(type), cudaMemcpyDeviceToHost));
            solid_neighbors = 0;
            for (int d = 0; d < this->num_dims; ++d)
                for (int side = -1; side <= 1; side += 2) {
                    int neighbor[max_dims] = {i, j, k};
                    neighbor[d] += side;
                    int marker;
                    CUDA_CHECK(cudaMemcpy(&marker, geometry.markers.data() + this->grid.storageIndex(neighbor[0], neighbor[1], neighbor[2]),
                                          sizeof(marker), cudaMemcpyDeviceToHost));
                    solid_neighbors += marker != 0;
                }
        }
        const int local_index[max_dims] = {i, j, k};
        int on_face = 0;
        for (int d = 0; d < this->num_dims; ++d) {
            const int global_index = local_index[d] + this->grid.offset[d];
            on_face |= global_index == 0 || global_index == this->grid.global_cells[d] - 1;
        }
        packed[0] = 1;
        for (int d = 0; d < max_dims; ++d) packed[1 + d] = local_index[d] + this->grid.offset[d];
        packed[4] = type + 1;
        packed[5] = solid_neighbors + 1;
        packed[6] = on_face;
    }
    allreduceSum(packed, 8);
    peak.found = packed[0] > 0.5;
    if (!peak.found) return peak;
    for (int d = 0; d < max_dims; ++d) {
        peak.index[d] = static_cast<int>(packed[1 + d] + 0.5);
        peak.position[d] = this->grid.origin[d] + (peak.index[d] + 0.5)*this->grid.spacing[d];
    }
    peak.cell_type = static_cast<int>(packed[4] + 0.5) - 1;
    peak.solid_neighbors = static_cast<int>(packed[5] + 0.5) - 1;
    peak.on_domain_face = packed[6] > 0.5;
    return peak;
}

StepDiagnostics FlowSolver::diagnostics(real dt) {
    StepDiagnostics result;
    const long count = this->grid.numOwned();
    const signed char* cell_types = this->immersed_boundaries.isActive() ? this->immersed_boundaries.geometry.cell_types.data() : nullptr;
    auto fill = [&](int quantity) {
        diagnosticValuesKernel<<<numBlocks(count), block_size>>>(this->grid, quantity, dt, this->velocity[0].data(), this->velocity[1].data(),
                                                                 this->velocity[2].data(), this->pressure.data(), this->rhs.data());
        CUDA_CHECK(cudaGetLastError());
    };
    fill(quantity_speed);
    result.speed = this->locatePeak(nullptr);
    fill(quantity_cfl);
    result.cfl = this->locatePeak(nullptr);
    divergenceKernel<<<numBlocks(count), block_size>>>(this->grid, this->faces[0].data(), this->faces[1].data(), this->faces[2].data(),
                                                       this->immersed_boundaries.geometry.cell_types.data(), 1.0, true, this->rhs.data());
    result.divergence = this->locatePeak(nullptr);
    fill(quantity_pressure);
    result.pressure = this->locatePeak(cell_types);
    return result;
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
