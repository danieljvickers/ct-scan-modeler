"""Case parameter definitions shared by the Python prototype and the CUDA solver."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Parameter:
    kind: type        # bool, int, float, or str
    default: object   # None means the parameter is required
    meaning: str


LOGICAL, INTEGER, REAL, STRING = bool, int, float, str

NUM_DIMS_MAX = 3
NUM_PATCHES_MAX = 10
NUM_IBS_MAX = 50

# scalar parameters, names follow MFC where the meaning carries over
SCALARS = {
    # logistics
    "run_time_info": Parameter(LOGICAL, False, "print solver progress at every save"),
    "case_dir": Parameter(STRING, ".", "directory holding case.py, set by the toolchain"),
    # computational domain
    "x_domain%beg": Parameter(REAL, None, "domain start in x before stretching"),
    "x_domain%end": Parameter(REAL, None, "domain end in x before stretching"),
    "y_domain%beg": Parameter(REAL, 0.0, "domain start in y before stretching"),
    "y_domain%end": Parameter(REAL, 0.0, "domain end in y before stretching"),
    "z_domain%beg": Parameter(REAL, 0.0, "domain start in z before stretching"),
    "z_domain%end": Parameter(REAL, 0.0, "domain end in z before stretching"),
    "m": Parameter(INTEGER, None, "number of cells in x minus one"),
    "n": Parameter(INTEGER, 0, "number of cells in y minus one"),
    "p": Parameter(INTEGER, 0, "number of cells in z minus one"),
    # time stepping
    "dt": Parameter(REAL, None, "time step size"),
    "t_step_start": Parameter(INTEGER, 0, "first time step"),
    "t_step_stop": Parameter(INTEGER, None, "last time step"),
    "t_step_save": Parameter(INTEGER, None, "time steps between field saves"),
    # simulation algorithm
    "num_patches": Parameter(INTEGER, 1, "number of initial condition patches"),
    "rk_single_projection": Parameter(LOGICAL, False, "project only at the last Runge-Kutta stage, Le & Moin (1991)"),
    "rk_order": Parameter(INTEGER, 1, "Runge-Kutta order 1, 2, or 3, each stage with Crank-Nicolson diffusion and a projection"),
    "poisson_solver": Parameter(INTEGER, 1, "1 = sparse direct LU, 2 = GMRES with a spectral preconditioner (uniform grid)"),
    "krylov_tolerance": Parameter(REAL, 1.0e-10, "relative residual at which GMRES stops"),
    "krylov_restart": Parameter(INTEGER, 30, "GMRES restart length"),
    "krylov_max_iterations": Parameter(INTEGER, 1000, "GMRES iteration limit before aborting"),
    "viscous": Parameter(LOGICAL, True, "include viscous diffusion"),
    # immersed boundaries
    "ib": Parameter(LOGICAL, False, "enable immersed boundaries"),
    "num_ibs": Parameter(INTEGER, 0, "number of immersed boundary patches"),
    "gp_layers": Parameter(INTEGER, 1, "neighbor search width that marks a solid cell as a ghost point"),
    # output
    "prim_vars_wrt": Parameter(LOGICAL, True, "write primitive variables at each save"),
    "ib_force_wrt": Parameter(LOGICAL, False, "write immersed boundary forces every step"),
    "bc_ramp_time": Parameter(REAL, 0.0, "time over which boundary pressures and inflow velocities grow linearly from zero, 0 applies them at once"),
    "ib_fill_faces": Parameter(INTEGER, 0, "fluid cells with at least this many solid face neighbors become solid, 0 keeps them all"),
    "ib_wall_upwind": Parameter(REAL, 0.0, "upwind fraction of convection on faces next to the immersed wall, 0 central, 1 first order upwind"),
    "ib_block_faces": Parameter(LOGICAL, False, "zero normal velocity on faces between fluid and solid cells, with zero flux pressure faces to match"),
    "weno_order": Parameter(INTEGER, 0, "convection face values, 0 central, 3 WENO3 (Jiang and Shu 1996)"),
    "weno_wall_only": Parameter(LOGICAL, True, "use WENO only on faces next to the immersed wall, central elsewhere"),
    "tiled_grid": Parameter(LOGICAL, False, "store only 8^3 tiles (16^2 in 2D) near the fluid and precondition with multigrid, on one GPU"),
    "diag_wrt": Parameter(LOGICAL, False, "write the location of the peak speed, CFL number, divergence, and pressure every step to D/diagnostics.dat"),
}

for direction in "xyz":
    SCALARS[f"stretch_{direction}"] = Parameter(LOGICAL, False, f"hyperbolic tangent stretching in {direction}")
    SCALARS[f"a_{direction}"] = Parameter(REAL, 0.0, f"stretching rate in {direction}")
    SCALARS[f"{direction}_a"] = Parameter(REAL, 0.0, f"start of the clustered region in {direction}")
    SCALARS[f"{direction}_b"] = Parameter(REAL, 0.0, f"end of the clustered region in {direction}")
    SCALARS[f"loops_{direction}"] = Parameter(INTEGER, 1, f"stretching passes in {direction}")
    for side in ("beg", "end"):
        SCALARS[f"bc_{direction}%{side}"] = Parameter(INTEGER, -3, f"boundary condition at the {direction} {side}")
        SCALARS[f"bc_{direction}%pres_{side}"] = Parameter(REAL, 0.0, "kinematic pressure held at an outflow (-3) boundary")
        for component in range(1, NUM_DIMS_MAX + 1):
            SCALARS[f"bc_{direction}%vel_in({component})"] = Parameter(REAL, 0.0, "Dirichlet inflow velocity")

SCALARS["fluid_pp(1)%Re(1)"] = Parameter(REAL, 0.0, "shear Reynolds number, the inverse kinematic viscosity")

# indexed families, e.g. patch_ib(2)%geometry
FAMILIES = {
    "patch_icpp": (NUM_PATCHES_MAX, {
        "geometry": Parameter(INTEGER, 3, "3 = rectangle, 7 = hardcoded 2D field chosen by hcid"),
        "hcid": Parameter(INTEGER, 0, "hardcoded field id: 201 = Gaussian vortex"),
        "radius": Parameter(REAL, 0.0, "hardcoded vortex radius R"),
        "vortex_velocity": Parameter(REAL, 0.0, "hardcoded vortex peak swirl velocity V at r = R"),
        "x_centroid": Parameter(REAL, 0.0, "patch center in x"),
        "y_centroid": Parameter(REAL, 0.0, "patch center in y"),
        "z_centroid": Parameter(REAL, 0.0, "patch center in z"),
        "length_x": Parameter(REAL, 0.0, "patch length in x"),
        "length_y": Parameter(REAL, 0.0, "patch length in y"),
        "length_z": Parameter(REAL, 0.0, "patch length in z"),
        "vel(1)": Parameter(REAL, 0.0, "initial x velocity"),
        "vel(2)": Parameter(REAL, 0.0, "initial y velocity"),
        "vel(3)": Parameter(REAL, 0.0, "initial z velocity"),
        "pres": Parameter(REAL, 0.0, "initial kinematic pressure"),
    }),
    "patch_ib": (NUM_IBS_MAX, {
        "geometry": Parameter(STRING, None, "STL path, relative to case_dir; a flat STL in 2D"),
        "slip": Parameter(LOGICAL, False, "slip wall, only no-slip is supported"),
        "model_threshold": Parameter(REAL, 0.5, "winding number above which a cell is solid"),
        "invert": Parameter(LOGICAL, False, "solid is everything outside the model, for flow inside a closed STL"),
        **{f"model_scale({k})": Parameter(REAL, 1.0, "model scale") for k in range(1, 4)},
        **{f"model_translate({k})": Parameter(REAL, 0.0, "model translation") for k in range(1, 4)},
        **{f"model_rotate({k})": Parameter(REAL, 0.0, "model rotation in radians") for k in range(1, 4)},
        "x_centroid": Parameter(REAL, 0.0, "torque reference point in x"),
        "y_centroid": Parameter(REAL, 0.0, "torque reference point in y"),
        "z_centroid": Parameter(REAL, 0.0, "torque reference point in z"),
    }),
}

FAMILY_PATTERN = re.compile(r"^(\w+)\((\d+)\)%(.+)$")


def lookup(name):
    """Return the Parameter for a case key, or None if it is not defined."""
    if name in SCALARS:
        return SCALARS[name]
    match = FAMILY_PATTERN.match(name)
    if match is None or match.group(1) not in FAMILIES:
        return None
    max_index, attributes = FAMILIES[match.group(1)]
    if not 1 <= int(match.group(2)) <= max_index:
        return None
    return attributes.get(match.group(3))


# parameters compiled in as constants by a case optimized build, changing any of them triggers a rebuild
CASE_OPTIMIZATION = ("m", "n", "p", "rk_order", "rk_single_projection", "poisson_solver", "viscous", "ib", "num_ibs", "gp_layers", "krylov_restart")
