# CT Scan Modeler

Turn CT data into STL surfaces and run incompressible flow past them with a ghost-cell immersed boundary method.
The flow solver will run in 2D and 3D on GPUs. This repository holds the toolchain, the 2D Python prototype that the CUDA solver will be checked against, and the start of the CUDA solver, which is a hello world for now.

## Contents

- [Layout](#layout)
- [Quick start](#quick-start)
- [Toolchain](#toolchain)
- [Numerical method](#numerical-method)
- [Case parameters](#case-parameters)
- [References](#references)

## Layout

| Path | Purpose |
|---|---|
| `ctm.sh` | Toolchain entry point: `./ctm.sh <run\|build\|test\|model>` |
| `toolchain/` | Case parameters, loader, and validator shared by both solvers; build and run commands; computer environments |
| `prototype/` | 2D Python solver, one module per MFC simulation module (`m_grid` → `grid.py`, `m_ibm` → `ibm.py`, ...) |
| `src/simulation/` | CUDA solver, a hello world for now |
| `examples/` | Case files; each `case.py` prints a JSON dictionary with MFC style keys |
| `tests/` | Unit tests for the prototype and toolchain |

## Quick start

```bash
python3 -m venv build/venv                                     # project environment
build/venv/bin/pip install -r toolchain/requirements.txt       # numpy, scipy, matplotlib, pytest
build/venv/bin/pip install -r toolchain/requirements-gpu.txt   # optional, cupy for CUDA 13
build/venv/bin/python -m prototype examples/2D_ibm_stl_cylinder/case.py
build/venv/bin/python -m pytest tests                          # unit tests
```

Case files take no command line arguments: every parameter lives in the case dictionary, computed at the top of `case.py` when it derives from other inputs. Output goes to `<case_dir>/D` unless you pass `--output_dir`.
Set `MFC_CUPY=1` to run on one GPU through CuPy, the same switch mfpython uses. If CuPy cannot be imported, the solver warns and falls back to NumPy.
Select the GPU with `CUDA_VISIBLE_DEVICES`.

## Toolchain

```bash
./ctm.sh run examples/2D_ibm_stl_cylinder/case.py -c wingtip -n 2                         # case optimized build, run on 2 GPUs
./ctm.sh run examples/2D_ibm_stl_cylinder/case.py -c w --no-case-optimization              # generic build
./ctm.sh build -c w                                                                         # generic build only
```

`./ctm.sh` creates `build/venv` from `toolchain/requirements.txt` on first use.

`run` does the following:
1. Validates the case with `toolchain/case_validator.py`, the same checks the prototype runs.
2. Builds the solver with CMake. The build is case optimized by default: the parameters in `CASE_OPTIMIZATION` (`toolchain/parameters.py`) are compiled in as `constexpr` values in `case_parameters.hpp`.
3. Writes the validated case to `simulation.json` in the case directory.
4. Launches one MPI rank per GPU with `mpirun -np <n>`, or with `srun` inside a Slurm allocation. Pass `--hostfile` for multinode runs outside Slurm.

While building, a single progress bar replaces the CMake and compiler output and clears when the build finishes. If a step fails, its full output is printed instead.

Each build configuration has its own directory, `build/staging/<hash>` and `build/install/<hash>`. The hash covers the case optimized parameters, the computer, and the build type, so changing `dt` or `t_stop` reuses an existing build, and changing `m` makes a new one. `build`, `test`, and `--no-case-optimization` use the generic build.

| Option | Default | Meaning |
|---|---|---|
| `-c`, `--computer` | current shell | Environment script from `toolchain/computers`, by name or alias (`wingtip`, `w`) |
| `-n`, `--num_gpus` | 1 | GPUs, one MPI rank each; ranks on a node take its GPUs round robin |
| `-j`, `--jobs` | 16 | Parallel build jobs |
| `--debug` | off | Debug build |
| `-v`, `--verbose` | off | Print the full CMake and compiler output instead of the progress bar |
| `--no-case-optimization` | off | Generic build |
| `--hostfile` | none | MPI hostfile for multinode runs outside Slurm |

### Regression tests

```bash
./ctm.sh test -c w                       # every test, generic build
./ctm.sh test -c w --only "no-slip"      # tests whose id or name matches
./ctm.sh test -c w --generate            # rewrite the golden files after an intended change
build/venv/bin/python -m pytest tests    # unit tests of the prototype and toolchain
```

Each test in `toolchain/test_cases.py` starts from an example's `case.py`, divides the cells per direction by its coarsening factor (4 by default), multiplies `dt` by the same factor to keep the CFL number, and runs 20 steps in `build/tests/<id>`. The id is a hash of the test name. The last save is reduced to the L1, L2, and max norms and the sum of every field, 64 probe values per field, and the final body forces. These values are compared with `tests/regression/<id>/golden.txt` using a relative tolerance of 1e-8, plus an absolute floor of 1e-9 times each entry's largest magnitude so round-off zeros, such as the lift of a symmetric body, pass. A test with `golden_from` must reproduce another test's golden file; the 2 GPU cylinder uses this to check that the decomposition does not change the answer.

| Test | Covers |
|---|---|
| 2D cylinder | geometry, ghost points, both implicit solves, forces |
| 2D cylinder on 2 GPUs | halo exchange and distributed transforms, against the 1 GPU golden |
| 2D cylinder without immersed boundary | domain boundaries and solvers alone |
| 2D cylinder with no-slip walls | the -16 boundary rules |

A computer is one shell script in `toolchain/computers` that exports the compilers, MPI, and `CTM_CUDA_ARCHITECTURES`, with a `# aliases:` comment line. `model` is a placeholder.

## Numerical method

### Why not a spectral method

Spectral methods get their accuracy from smooth, periodic fields.
An immersed body puts a discontinuity in the velocity gradient inside the domain, and inflow and outflow boundaries are not periodic, so the spectral advantage is lost to Gibbs oscillations.
Their global transforms also do not mix with the local stencils of a ghost-cell method.
A second order finite volume scheme matches the second order accuracy of the ghost-cell IBM itself [1, 2], uses only nearest neighbor stencils that port directly to CUDA, and keeps the cell-centered layout of MFC.

### Scheme

- **Grid**: collocated cell-centered finite volume, uniform or MFC tanh stretching.
- **Time integration**: Runge-Kutta stages in the form of Le & Moin (1991): explicit convection, Crank-Nicolson diffusion, and a projection each stage. Only rk1 is implemented; it is stable for central convection while dt ≤ 2ν/U², which holds for the low Reynolds number cases here.
- **Pressure**: incremental pressure correction projection [3, 4]. Face velocities are kept separately and made exactly divergence free, the collocated approach of [5] used with immersed boundaries in [6]. A Rhie-Chow style term on interior fluid faces prevents pressure checkerboarding.
- **Linear solves**, selected by `poisson_solver`:
  - `1`: the velocity Helmholtz and pressure Poisson matrices are LU factorized once with SuperLU. This works on stretched grids but is slow on the GPU.
  - `2`: GMRES [9] on the immersed boundary system, preconditioned by an exact sine/cosine transform solve of the no-body operator [10]. This needs a uniform grid and runs on cuFFT through CuPy.

### Immersed boundaries

The scheme follows MFC's `m_ibm`:

- **Solid cells**: a cell is solid if its center has an STL winding number above `model_threshold` [7]. A solid cell with fluid within `gp_layers` cells is a ghost point.
- **Image points**: each image point mirrors its ghost point across the nearest STL edge.
- **Interpolation**: image point values use bilinear weights [2] when all four stencil cells are fluid, and inverse distance squared weights on the fluid cells otherwise [8].
- **Velocity**: mirrored, `u_ghost = -u_image`, for a stationary no-slip wall. This relation goes into the implicit velocity solve.
- **Pressure**: mirrored, `p_ghost = p_image`, which imposes a zero normal pressure gradient inside the Poisson solve.
- **Forces**: the stress divergence summed over solid cells, as in MFC's `s_compute_ib_forces`.

## Case parameters

Keys follow MFC wherever the meaning carries over. The full list with defaults is in `toolchain/parameters.py`. Unknown keys are rejected.

| Variable Name | Default | Meaning |
|---|---|---|
| `x_domain%beg`, `x_domain%end`, `m` | required | Domain before stretching and number of cells minus one |
| `stretch_x`, `a_x`, `x_a`, `x_b`, `loops_x` | `F`, 0, 0, 0, 1 | MFC tanh stretching, uniform between `x_a` and `x_b` |
| `dt`, `t_step_stop`, `t_step_save` | required | Fixed time step, last step, save interval |
| `poisson_solver` | 1 | 1 = sparse direct LU, 2 = spectral preconditioned GMRES (uniform grid) |
| `fluid_pp(1)%Re(1)` | 0 | Inverse kinematic viscosity, as in MFC |
| `bc_x%beg` | -3 | -3 outflow (`p = 0`), -15 slip wall, -16 no-slip wall, -17 inflow at `bc_x%vel_in(i)` |
| `patch_ib(i)%geometry` | required | STL path relative to the case directory; a flat STL in 2D |
| `patch_ib(i)%model_threshold` | 0.5 | Winding number above which a cell is solid |
| `ib_force_wrt` | `F` | Write per-step body forces to `D/ib_forces.dat` |

Pressure is kinematic (pressure over density) and density is 1.

### References

1. Tseng, Y.-H. & Ferziger, J. H. (2003). A ghost-cell immersed boundary method for flow in complex geometry. *J. Comput. Phys.* 192, 593–623. doi:10.1016/j.jcp.2003.07.024
2. Mittal, R., Dong, H., Bozkurttas, M., Najjar, F. M., Vargas, A. & von Loebbecke, A. (2008). A versatile sharp interface immersed boundary method for incompressible flows with complex boundaries. *J. Comput. Phys.* 227, 4825–4852. doi:10.1016/j.jcp.2008.01.028
3. Kim, J. & Moin, P. (1985). Application of a fractional-step method to incompressible Navier-Stokes equations. *J. Comput. Phys.* 59, 308–323. doi:10.1016/0021-9991(85)90148-2
4. Van Kan, J. (1986). A second-order accurate pressure-correction scheme for viscous incompressible flow. *SIAM J. Sci. Stat. Comput.* 7, 870–891. doi:10.1137/0907059
5. Zang, Y., Street, R. L. & Koseff, J. R. (1994). A non-staggered grid, fractional step method for time-dependent incompressible Navier-Stokes equations in curvilinear coordinates. *J. Comput. Phys.* 114, 18–33. doi:10.1006/jcph.1994.1146
6. Ye, T., Mittal, R., Udaykumar, H. S. & Shyy, W. (1999). An accurate Cartesian grid method for viscous incompressible flows with complex immersed boundaries. *J. Comput. Phys.* 156, 209–240. doi:10.1006/jcph.1999.6356
7. Jacobson, A., Kavan, L. & Sorkine-Hornung, O. (2013). Robust inside-outside segmentation using generalized winding numbers. *ACM Trans. Graph.* 32(4), 33. doi:10.1145/2461912.2461916
8. Chaudhuri, A., Hadjadj, A. & Chinnayya, A. (2011). On the use of immersed boundary methods for shock/obstacle interactions. *J. Comput. Phys.* 230, 1731–1748. doi:10.1016/j.jcp.2010.11.016
9. Saad, Y. & Schultz, M. H. (1986). GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems. *SIAM J. Sci. Stat. Comput.* 7, 856–869. doi:10.1137/0907058
10. Makhoul, J. (1980). A fast cosine transform in one and two dimensions. *IEEE Trans. Acoust., Speech, Signal Process.* 28, 27–34. doi:10.1109/TASSP.1980.1163351
