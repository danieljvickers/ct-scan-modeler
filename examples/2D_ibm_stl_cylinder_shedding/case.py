#!/usr/bin/env python3
# vortex shedding behind a circular cylinder at Re = 200; Taira & Colonius (2007), as tabulated in later reviews,
# report St = 0.195, C_D = 1.34 +/- 0.047, C_L = +/- 0.68, and Williamson (1996) measures St = 0.197
import json
import math

# physical inputs
diameter = 1.0  # STL disk diameter
free_stream_velocity = 1.0
Re = 200.0
kick_velocity = 0.25  # transverse velocity behind the body at t = 0, starts the shedding without waiting on round-off

# uniform grid and time stepping in convective units, rk1 with central convection needs dt below 2 nu/|u|^2
cells_per_diameter = 32
x_beg, x_end = -15.0, 35.0
y_beg, y_end = -15.0, 15.0
cell_width = diameter/cells_per_diameter
save_interval = 0.25  # about 20 saves per shedding period
max_speed = 1.6*free_stream_velocity
steps_per_save = math.ceil(save_interval/(0.5*2.0*diameter/Re/max_speed**2))
dt = save_interval/steps_per_save
t_stop = 150.0
t_step_stop = int(round(t_stop/dt))

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
    "t_step_save": steps_per_save,
    # Simulation Algorithm Parameters
    "num_patches": 2,
    "poisson_solver": 2,  # GMRES with a spectral preconditioner
    "krylov_tolerance": 1.0e-10,
    "viscous": "T",
    "bc_x%beg": -17,  # Dirichlet inflow
    "bc_x%vel_in(1)": free_stream_velocity,
    "bc_x%vel_in(2)": 0.0,
    "bc_x%end": -3,  # outflow
    "bc_y%beg": -15,  # slip wall
    "bc_y%end": -15,
    # Patch 1: uniform free stream
    "patch_icpp(1)%geometry": 3,
    "patch_icpp(1)%x_centroid": 0.0,
    "patch_icpp(1)%y_centroid": 0.0,
    "patch_icpp(1)%length_x": 1000.0*diameter,
    "patch_icpp(1)%length_y": 1000.0*diameter,
    "patch_icpp(1)%vel(1)": free_stream_velocity,
    "patch_icpp(1)%vel(2)": 0.0,
    "patch_icpp(1)%pres": 0.0,
    # Patch 2: transverse kick in the upper near wake
    "patch_icpp(2)%geometry": 3,
    "patch_icpp(2)%x_centroid": 2.0*diameter,
    "patch_icpp(2)%y_centroid": 0.5*diameter,
    "patch_icpp(2)%length_x": 2.0*diameter,
    "patch_icpp(2)%length_y": 1.0*diameter,
    "patch_icpp(2)%vel(1)": free_stream_velocity,
    "patch_icpp(2)%vel(2)": kick_velocity,
    "patch_icpp(2)%pres": 0.0,
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
