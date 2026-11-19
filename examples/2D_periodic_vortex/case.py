#!/usr/bin/env python3
# shielded Gaussian vortex advected diagonally through a periodic box for one period, the exact solution is the
# initial vortex diffused by the axisymmetric heat equation: u_theta = V e^(1/2) (R^3/s^4) r exp(-r^2/(2 s^2)), s^2 = R^2 + 2 nu t
import json
import math

# physical inputs
vortex_radius = 1.0
vortex_velocity = 0.5    # peak swirl at r = R
free_stream = (1.0, 1.0)  # diagonal, so both periodic directions wrap
kinematic_viscosity = 0.002
box_length = 10.0*vortex_radius  # swirl at the box edge is below 1e-5

# grid and time stepping, rk1 with central convection needs dt below 2 nu/|u|^2
num_cells = 128
period = box_length/free_stream[0]
max_speed_squared = (math.hypot(*free_stream) + vortex_velocity)**2
dt = 0.5*2.0*kinematic_viscosity/max_speed_squared
t_step_stop = int(math.ceil(period/dt))
dt = period/t_step_stop  # land exactly on one period
t_step_save = t_step_stop//20

print(json.dumps({
    # Logistics
    "run_time_info": "T",
    # Computational Domain Parameters
    "x_domain%beg": -0.5*box_length,
    "x_domain%end": 0.5*box_length,
    "y_domain%beg": -0.5*box_length,
    "y_domain%end": 0.5*box_length,
    "m": num_cells - 1,
    "n": num_cells - 1,
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
    "bc_x%beg": -1,  # periodic
    "bc_x%end": -1,
    "bc_y%beg": -1,
    "bc_y%end": -1,
    # Patch: Gaussian vortex on the free stream
    "patch_icpp(1)%geometry": 7,  # hardcoded 2D field
    "patch_icpp(1)%hcid": 201,    # Gaussian vortex
    "patch_icpp(1)%x_centroid": 0.0,
    "patch_icpp(1)%y_centroid": 0.0,
    "patch_icpp(1)%radius": vortex_radius,
    "patch_icpp(1)%vortex_velocity": vortex_velocity,
    "patch_icpp(1)%vel(1)": free_stream[0],
    "patch_icpp(1)%vel(2)": free_stream[1],
    "patch_icpp(1)%pres": 0.0,
    # Fluid Physical Parameters
    "fluid_pp(1)%Re(1)": 1.0/kinematic_viscosity,
    # Output
    "prim_vars_wrt": "T",
}))
