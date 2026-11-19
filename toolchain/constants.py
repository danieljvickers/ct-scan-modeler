"""Integer flags shared by case files, the prototype, and the CUDA solver; values match MFC where the meaning carries over."""

# boundary conditions
BC_GHOST_EXTRAP = -3   # outflow: zero normal velocity gradient, zero pressure
BC_SLIP_WALL = -15     # zero normal velocity, zero tangential velocity gradient
BC_NO_SLIP_WALL = -16  # zero velocity
BC_DIRICHLET = -17     # inflow at bc_%vel_in, zero pressure gradient
SUPPORTED_BCS = (BC_GHOST_EXTRAP, BC_SLIP_WALL, BC_NO_SLIP_WALL, BC_DIRICHLET)

# initial condition patch geometries
ICPP_RECTANGLE = 3

# pressure solvers
POISSON_SPARSE_LU = 1
POISSON_SPECTRAL_KRYLOV = 2
