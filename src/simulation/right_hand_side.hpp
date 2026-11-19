// right hand side and time stepping: incremental pressure correction on a collocated grid with face velocities
// projection, Kim & Moin (1985) and Van Kan (1986); Runge-Kutta stages with Crank-Nicolson diffusion, Le & Moin (1991);
// collocated face velocities, Zang, Street & Koseff (1994)
#ifndef CTM_RIGHT_HAND_SIDE_HPP
#define CTM_RIGHT_HAND_SIDE_HPP

#include <array>
#include <memory>
#include <vector>

#include "ibm.hpp"
#include "linear_solvers.hpp"

// one Runge-Kutta stage: u* = u - dt*(gamma*N + zeta*N_previous) - alpha*dt*grad(p) + alpha*dt*nu*(L u* + L u)/2
struct StageCoefficients {
    real gamma;  // weight of this stage's convection
    real zeta;   // weight of the previous stage's convection
    real alpha;  // stage time fraction for pressure and diffusion
};

constexpr StageCoefficients rk1_stages[] = {{1.0, 0.0, 1.0}};

class FlowSolver {
public:
    FlowSolver(const Parameters& parameters, const GridView& grid, ImmersedBoundaries& immersed_boundaries);

    void initialize();
    void advance();  // one time step through every stage

    real maxDivergence();
    bool isFinite();
    std::vector<real> forces();
    std::array<int, num_field_kinds> lastIterations() const { return this->iterations; }

    // fields with halos, the w component stays zero in 2D
    DeviceArray<real> velocity[max_dims];
    DeviceArray<real> pressure;

private:
    void stage(int stage_index);
    void project(real stage_dt);
    void fillVelocityHalos(DeviceArray<real>* fields);
    void applyOperator(const real* x, real* y, FieldKind kind, real shift, real scale, real ghost_sign);

    const Parameters& parameters;
    GridView grid;
    ImmersedBoundaries& immersed_boundaries;
    std::vector<StageCoefficients> stages;
    int num_dims;

    HaloExchange exchange;
    SlabTranspose transpose;
    LineTransforms transforms;
    Gmres gmres;
    HaloRule rules[num_field_kinds];
    HaloRule homogeneous_rules[num_field_kinds];
    std::vector<std::unique_ptr<SpectralSolver>> helmholtz_preconditioners;  // [stage][component]
    std::unique_ptr<SpectralSolver> poisson_preconditioner;

    DeviceArray<real> faces[max_dims];                  // face normal velocity, face i of a cell stored at that cell
    DeviceArray<real> predicted[max_dims];              // tentative velocity with halos
    DeviceArray<real> convection[max_dims];             // owned compact
    DeviceArray<real> previous_convection[max_dims];    // owned compact, for zeta
    DeviceArray<real> boundary_offsets[max_dims];       // laplacian of the boundary values, owned compact
    DeviceArray<real> rhs, solution, pressure_correction, operator_input, correction_field, reduction;
    std::array<int, num_field_kinds> iterations = {0, 0, 0, 0};
};

#endif
