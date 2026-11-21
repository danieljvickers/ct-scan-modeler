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
    bc_periodic = -1,             // wraps to the opposite side
    bc_ghost_extrapolation = -3,  // outflow: zero normal velocity gradient, pressure held at boundary_pressure
    bc_slip_wall = -15,           // zero normal velocity, zero tangential gradient
    bc_no_slip_wall = -16,        // zero velocity
    bc_dirichlet = -17,           // inflow at bc_%vel_in, zero pressure gradient
};

// initial condition patch geometries and hardcoded fields, values match the toolchain constants
enum PatchGeometry : int { patch_rectangle = 3, patch_hardcoded_2d = 7 };
enum HardcodedField : int { hcid_gaussian_vortex = 201 };

struct InitialPatch {
    int geometry;
    int hcid;
    real radius;           // Gaussian vortex radius R
    real vortex_velocity;  // Gaussian vortex peak swirl V at r = R
    real centroid[max_dims];
    real length[max_dims];
    real velocity[max_dims];
    real pressure;
};

struct ImmersedPatch {
    std::string stl_path;
    real threshold;  // winding number above which a cell is solid
    bool invert = false;  // solid is everything outside the model, for flow inside a closed surface
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
    int rk_order;  // Runge-Kutta stages per step, each with Crank-Nicolson diffusion and a projection
    bool rk_single_projection;  // project only at the last stage, Le & Moin (1991)

    bool viscous;
    real kinematic_viscosity;  // 1/fluid_pp(1)%Re(1), density is 1
    real krylov_tolerance;
    int krylov_restart, krylov_max_iterations;

    int boundary[max_dims][2];             // [direction][begin, end]
    real inflow_velocity[max_dims][max_dims];  // [direction][component] for Dirichlet boundaries
    real boundary_pressure[max_dims][2];       // [direction][side] kinematic pressure at outflow (-3) boundaries

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
