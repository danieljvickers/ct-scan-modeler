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
#include "multigrid.hpp"

// one Runge-Kutta stage: u* = u - dt*(gamma*N + zeta*N_previous) - alpha*dt*grad(p) + alpha*dt*nu*(L u* + L u)/2
struct StageCoefficients {
    real gamma;  // weight of this stage's convection
    real zeta;   // weight of the previous stage's convection
    real alpha;  // stage time fraction for pressure and diffusion
};

// rk2 is the explicit midpoint rule, rk3 the low storage scheme of Wray as used by Le & Moin (1991)
constexpr StageCoefficients rk1_stages[] = {{1.0, 0.0, 1.0}};
constexpr StageCoefficients rk2_stages[] = {{0.5, 0.0, 0.5}, {1.0, -0.5, 0.5}};
constexpr StageCoefficients rk3_stages[] = {{8.0/15.0, 0.0, 8.0/15.0}, {5.0/12.0, -17.0/60.0, 2.0/15.0}, {3.0/4.0, -5.0/12.0, 1.0/3.0}};

std::vector<StageCoefficients> rungeKuttaStages(int rk_order);

// where a field is largest and what kind of cell that is, in global indices
struct FieldPeak {
    real value = 0.0;
    bool found = false;
    int index[3] = {-1, -1, -1};
    real position[3] = {0.0, 0.0, 0.0};
    int cell_type = -1;        // 0 fluid, 1 ghost, 2 solid interior, -1 unknown
    int solid_neighbors = -1;  // of the six face neighbors
    int on_domain_face = 0;
};

struct StepDiagnostics {
    FieldPeak speed, cfl, divergence, pressure;
};

class FlowSolver {
public:
    FlowSolver(const Parameters& parameters, const GridView& grid, ImmersedBoundaries& immersed_boundaries);

    void initialize();
    void advance();  // one time step through every stage

    real maxDivergence();
    StepDiagnostics diagnostics(real dt);  // peak speed, convective CFL number, divergence, and pressure, with their cells
    bool isFinite();
    std::vector<real> forces();
    std::array<int, num_field_kinds> lastIterations() const { return this->iterations; }

    // fields with halos, the w component stays zero in 2D
    DeviceArray<real> velocity[max_dims];
    DeviceArray<real> pressure;

private:
    void stage(int stage_index);
    FieldPeak locatePeak(const signed char* cell_types);  // largest |value| of this->rhs
    void project(real stage_dt);
    void rampBoundaryValues(real time);  // scale the boundary values for bc_ramp_time, then refresh the boundary offsets
    void computeBoundaryOffsets();
    const int* blockMarkers() const {  // solid markers when faces next to the wall are blocked, otherwise null
        return this->parameters.ib_block_faces && this->immersed_boundaries.isActive() ? this->immersed_boundaries.geometry.markers.data() : nullptr;
    }
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
    HaloRule full_rules[num_field_kinds];  // boundary values at full strength, before any ramp
    real time = 0.0;
    std::unique_ptr<MultigridHierarchy> multigrid;  // the preconditioners' grids when the grid is tiled
    std::vector<std::unique_ptr<Preconditioner>> helmholtz_preconditioners;  // [stage][component]
    std::unique_ptr<Preconditioner> poisson_preconditioner;

    DeviceArray<real> faces[max_dims];                  // face normal velocity, face i of a cell stored at that cell
    DeviceArray<real> predicted[max_dims];              // tentative velocity with halos
    DeviceArray<real> convection[max_dims];             // owned vector
    DeviceArray<real> previous_convection[max_dims];    // owned vector, for zeta
    DeviceArray<real> boundary_offsets[max_dims];       // laplacian of the boundary values, owned vector
    DeviceArray<real> rhs, solution, pressure_correction, operator_input, correction_field, reduction;
    std::array<int, num_field_kinds> iterations = {0, 0, 0, 0};
};

#endif
