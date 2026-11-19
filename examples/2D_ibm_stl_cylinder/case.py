#!/usr/bin/env python3
# steady flow past a circular cylinder at Re = 40, compared with Taira & Colonius (2007)
import json

# physical inputs
diameter = 1.0  # STL disk diameter
free_stream_velocity = 1.0
Re = 40.0

# uniform grid and time stepping in convective units
cells_per_diameter = 32
cfl = 0.4
t_stop = 80.0
x_beg, x_end = -20.0, 40.0
y_beg, y_end = -20.0, 20.0
cell_width = diameter/cells_per_diameter
dt = cfl*cell_width/free_stream_velocity
t_step_stop = int(round(t_stop/dt))
t_step_save = t_step_stop//20

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
    "dt": dt,
    "t_step_start": 0,
    "t_step_stop": t_step_stop,
    "t_step_save": t_step_save,
    # Simulation Algorithm Parameters
    "num_patches": 1,
    "poisson_solver": 2,  # GMRES with a spectral preconditioner
    "krylov_tolerance": 1.0e-10,
    "viscous": "T",
    "bc_x%beg": -17,  # Dirichlet inflow
    "bc_x%vel_in(1)": free_stream_velocity,
    "bc_x%vel_in(2)": 0.0,
    "bc_x%end": -3,  # outflow
    "bc_y%beg": -15,  # slip wall
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
    "patch_ib(1)%geometry": "cylinder.stl",
    "patch_ib(1)%slip": "F",
    "patch_ib(1)%model_threshold": 0.5,
    # Fluid Physical Parameters
    "fluid_pp(1)%Re(1)": Re/(diameter*free_stream_velocity),
    # Output
    "prim_vars_wrt": "T",
    "ib_force_wrt": "T",
}))
