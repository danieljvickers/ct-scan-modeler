#!/usr/bin/env python3
# steady flow past a circular cylinder at Re = 20 or 40, compared with Taira & Colonius (2007)
import argparse
import json

parser = argparse.ArgumentParser(description="2D incompressible flow past an STL cylinder")
parser.add_argument("--Re", type=float, default=40.0, help="Reynolds number based on diameter")
parser.add_argument("--cells_per_diameter", type=int, default=32, help="uniform cells across the diameter near the body")
parser.add_argument("--t_stop", type=float, default=80.0, help="end time in convective units D/U")
parser.add_argument("--cfl", type=float, default=0.4, help="free stream CFL number")
parser.add_argument("--poisson_solver", type=int, default=2, choices=[1, 2], help="1 = LU on a stretched grid, 2 = spectral Krylov on a uniform grid")
arguments, _ = parser.parse_known_args()

# physical inputs
diameter = 1.0         # STL disk diameter
free_stream_velocity = 1.0
Re = arguments.Re

# domain: uniform for the spectral solver, else stretched from the near field out to about [-17, 35] x [-20, 20]
is_uniform = arguments.poisson_solver == 2
x_beg, x_end = (-20.0, 40.0) if is_uniform else (-5.5, 9.0)
y_beg, y_end = (-20.0, 20.0) if is_uniform else (-5.0, 5.0)
stretch = "F" if is_uniform else "T"
cell_width = diameter/arguments.cells_per_diameter

# time stepping in convective units
dt = arguments.cfl*cell_width/free_stream_velocity
t_step_stop = int(round(arguments.t_stop/dt))
t_step_save = max(1, int(round(t_step_stop/20)))

print(json.dumps({
    # Logistics
    "run_time_info": "T",
    # Computational Domain Parameters
    "x_domain%beg": x_beg,
    "x_domain%end": x_end,
    "y_domain%beg": y_beg,
    "y_domain%end": y_end,
    "m": int(round((x_end - x_beg)/cell_width)) - 1,
    "n": int(round((y_end - y_beg)/cell_width)) - 1,
    "p": 0,
    "stretch_x": stretch,
    "a_x": 20.0,
    "x_a": -1.5*diameter,
    "x_b": 4.5*diameter,  # keeps the Re = 40 recirculation region uniform
    "loops_x": 2,
    "stretch_y": stretch,
    "a_y": 20.0,
    "y_a": -1.5*diameter,
    "y_b": 1.5*diameter,
    "loops_y": 2,
    "dt": dt,
    "t_step_start": 0,
    "t_step_stop": t_step_stop,
    "t_step_save": t_step_save,
    # Simulation Algorithm Parameters
    "num_patches": 1,
    "num_fluids": 1,
    "time_stepper": 1,    # AB2 convection, Crank-Nicolson diffusion
    "poisson_solver": arguments.poisson_solver,
    "krylov_tolerance": 1.0e-10,
    "viscous": "T",
    "bc_x%beg": -17,      # Dirichlet inflow
    "bc_x%vel_in(1)": free_stream_velocity,
    "bc_x%vel_in(2)": 0.0,
    "bc_x%end": -3,       # outflow
    "bc_y%beg": -15,      # slip wall
    "bc_y%end": -15,
    # Patch: uniform free stream
    "patch_icpp(1)%geometry": 3,
    "patch_icpp(1)%x_centroid": 0.0,
    "patch_icpp(1)%y_centroid": 0.0,
    "patch_icpp(1)%length_x": 1000.0*diameter,
    "patch_icpp(1)%length_y": 1000.0*diameter,
    "patch_icpp(1)%vel(1)": free_stream_velocity,
    "patch_icpp(1)%vel(2)": 0.0,
    "patch_icpp(1)%pres": 0.0,
    # Immersed Boundary: STL cylinder
    "ib": "T",
    "num_ibs": 1,
    "ibm_ip_interpolation": 2,  # bilinear, inverse distance fallback near solids
    "patch_ib(1)%geometry": 5,  # 2D STL
    "patch_ib(1)%model_id": 1,
    "patch_ib(1)%slip": "F",
    "num_stl_models": 1,
    "stl_models(1)%model_filepath": "cylinder.stl",
    "stl_models(1)%model_threshold": 0.5,
    # Fluid Physical Parameters
    "fluid_pp(1)%Re(1)": Re/(diameter*free_stream_velocity),
    # Output
    "prim_vars_wrt": "T",
    "ib_force_wrt": "T",
}))
