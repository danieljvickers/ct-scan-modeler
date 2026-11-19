// case parameters read from the flat input file the toolchain writes, following MFC's m_global_parameters
#ifndef CTM_GLOBAL_PARAMETERS_HPP
#define CTM_GLOBAL_PARAMETERS_HPP

#include <string>
#include <vector>

#include "common.hpp"

#ifdef CTM_CASE_OPTIMIZATION
#include "case_parameters.hpp"
#endif

// boundary condition flags, values match MFC
enum BoundaryCondition : int {
    bc_ghost_extrapolation = -3,  // outflow: zero normal velocity gradient, zero pressure
    bc_slip_wall = -15,           // zero normal velocity, zero tangential gradient
    bc_no_slip_wall = -16,        // zero velocity
    bc_dirichlet = -17,           // inflow at bc_%vel_in, zero pressure gradient
};

struct InitialPatch {
    real centroid[max_dims];
    real length[max_dims];
    real velocity[max_dims];
    real pressure;
};

struct ImmersedPatch {
    std::string stl_path;
    real threshold;  // winding number above which a cell is solid
    real scale[max_dims];
    real translate[max_dims];
    real rotate[max_dims];  // radians
};

struct Parameters {
    std::string case_dir;
    bool run_time_info;

    int num_dims;
    int global_cells[max_dims];  // m + 1, n + 1, p + 1
    real domain_begin[max_dims];
    real domain_end[max_dims];

    real dt;
    int t_step_start, t_step_stop, t_step_save;

    bool viscous;
    real kinematic_viscosity;  // 1/fluid_pp(1)%Re(1), density is 1
    real krylov_tolerance;
    int krylov_restart, krylov_max_iterations;

    int boundary[max_dims][2];             // [direction][begin, end]
    real inflow_velocity[max_dims][max_dims];  // [direction][component] for Dirichlet boundaries

    std::vector<InitialPatch> initial_patches;
    bool ib;
    int gp_layers;
    std::vector<ImmersedPatch> immersed_patches;

    bool prim_vars_wrt;
    bool ib_force_wrt;
};

Parameters readParameters(const std::string& path);

// a case optimized binary must be run with the case it was built for
void checkCaseOptimization(const Parameters& parameters);

#endif
