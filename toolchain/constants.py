"""Integer flags shared by case files, the prototype, and the CUDA solver; values match MFC where the meaning carries over."""

# boundary conditions
BC_PERIODIC = -1       # wraps to the opposite side, both sides of a direction must be periodic
BC_GHOST_EXTRAP = -3   # outflow: zero normal velocity gradient, zero pressure
BC_SLIP_WALL = -15     # zero normal velocity, zero tangential velocity gradient
BC_NO_SLIP_WALL = -16  # zero velocity
BC_DIRICHLET = -17     # inflow at bc_%vel_in, zero pressure gradient
SUPPORTED_BCS = (BC_PERIODIC, BC_GHOST_EXTRAP, BC_SLIP_WALL, BC_NO_SLIP_WALL, BC_DIRICHLET)

# initial condition patch geometries
ICPP_RECTANGLE = 3
ICPP_HARDCODED_2D = 7  # analytic 2D field chosen by patch_icpp(i)%hcid, as in MFC
HCID_GAUSSIAN_VORTEX = 201  # shielded vortex u_theta = V (r/R) exp((1 - r^2/R^2)/2) on the patch velocity

# pressure solvers
POISSON_SPARSE_LU = 1
POISSON_SPECTRAL_KRYLOV = 2
