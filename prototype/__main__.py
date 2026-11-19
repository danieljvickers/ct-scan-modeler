"""Run a case with the 2D Python prototype: python -m prototype path/to/case.py [case arguments]."""

import argparse
import os
import time

from prototype.backend import USING_GPU, xp
from prototype.boundary_conditions import get_halos
from prototype.checker import check_case
from prototype.data_output import DataOutput
from prototype.grid import Grid
from prototype.ib_patches import compute_ib_markers, load_models
from prototype.ibm import ImmersedBoundaries
from prototype.initial_condition import initial_condition
from prototype.operators import Operators
from prototype.time_steppers import FractionalStepSolver
from toolchain.case import load_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_path", help="path to case.py")
    parser.add_argument("--output_dir", default=None, help="output directory, defaults to <case_dir>/D")
    arguments, case_arguments = parser.parse_known_args()

    case = load_case(arguments.case_path, case_arguments)
    check_case(case)
    output_dir = arguments.output_dir or os.path.join(case["case_dir"], "D")

    # set up the grid, bodies, and operators
    setup_start = time.perf_counter()
    grid = Grid(case)
    immersed_boundaries = None
    if case["ib"]:
        models = load_models(case)
        immersed_boundaries = ImmersedBoundaries(case, grid, compute_ib_markers(case, grid, models), models)
        num_bilinear = immersed_boundaries.ghost_points.is_bilinear.sum()
        print(f"INFO :: {len(immersed_boundaries.ghost_points)} ghost points, {num_bilinear} bilinear and the rest inverse distance")
    halos = get_halos(case)
    solver = FractionalStepSolver(case, grid, Operators(grid, halos), immersed_boundaries, halos)
    print(f"INFO :: {'GPU (cupy)' if USING_GPU else 'CPU (numpy)'}, {grid.shape[0]} x {grid.shape[1]} cells, set up in {time.perf_counter() - setup_start:.1f} s")

    output = DataOutput(case, grid, immersed_boundaries, output_dir)
    velocities, pressure = initial_condition(case, grid)
    velocities, face_velocities, pressure = solver.initialize(velocities, pressure)
    output.write_fields(case["t_step_start"], velocities, pressure)

    # march in time
    loop_start = time.perf_counter()
    for step in range(case["t_step_start"] + 1, case["t_step_stop"] + 1):
        velocities, face_velocities, pressure = solver.step(velocities, face_velocities, pressure)
        if case["ib"] and case["ib_force_wrt"]:
            output.write_forces(step, solver.forces(velocities, pressure))
        if step % case["t_step_save"] == 0:
            assert bool(xp.isfinite(velocities["u"]).all()), f"ERROR :: solution diverged at step {step}"
            output.write_fields(step, velocities, pressure)
            output.flush_forces()
            if case["run_time_info"]:
                seconds_per_step = (time.perf_counter() - loop_start)/(step - case["t_step_start"])
                iterations = "/".join(str(count) for count in solver.last_iterations())
                print(f"INFO :: step {step}/{case['t_step_stop']}, time {step*case['dt']:.4f}, "
                      f"max |div u| {solver.max_fluid_divergence(face_velocities):.2e}, "
                      f"u/v/p iterations {iterations}, {seconds_per_step:.3f} s/step")
    output.close()


if __name__ == "__main__":
    main()
