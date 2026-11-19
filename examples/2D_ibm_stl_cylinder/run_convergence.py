#!/usr/bin/env python3
"""Grid convergence of drag and wake length at Re = 20 and 40 against Taira & Colonius (2007)."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# math and scientific libraries
import numpy as np

# plotting libraries
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CASE_DIR, "..", ".."))

# steady wake length L/D and drag coefficient, Taira & Colonius (2007)
REFERENCE = {20: {"drag_coefficient": 2.06, "wake_length": 0.94}, 40: {"drag_coefficient": 1.54, "wake_length": 2.30}}
DIAMETER = 1.0
FREE_STREAM_VELOCITY = 1.0
AVERAGE_WINDOW = 0.05  # fraction of the run averaged for the final drag


def run_directory(reynolds_number, cells_per_diameter, poisson_solver):
    return os.path.join(CASE_DIR, "convergence", f"Re{reynolds_number}_N{cells_per_diameter}_solver{poisson_solver}")


def run_case(reynolds_number, cells_per_diameter, poisson_solver, t_stop, gpu):
    """Run one case, on the given GPU through CuPy or on the CPU when gpu is None."""
    output_dir = run_directory(reynolds_number, cells_per_diameter, poisson_solver)
    os.makedirs(output_dir, exist_ok=True)
    command = [sys.executable, "-u", "-m", "prototype", os.path.join(CASE_DIR, "case.py"), "--output_dir", output_dir,
               "--Re", str(reynolds_number), "--cells_per_diameter", str(cells_per_diameter), "--t_stop", str(t_stop),
               "--poisson_solver", str(poisson_solver)]
    environment = dict(os.environ, MFC_CUPY="0" if gpu is None else "1")
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with open(os.path.join(output_dir, "log.txt"), "w") as log_file:
        subprocess.run(command, cwd=REPO_ROOT, env=environment, stdout=log_file, stderr=subprocess.STDOUT, check=True)


def wake_length(output_dir):
    """Distance from the rear of the cylinder to where the centerline velocity turns positive, in diameters."""
    grid = np.load(os.path.join(output_dir, "grid.npz"))
    last_field = sorted(name for name in os.listdir(output_dir) if name.startswith("fields_"))[-1]
    u = np.load(os.path.join(output_dir, last_field))["u"]

    # interpolate to y = 0 between the two bracketing rows
    upper = np.searchsorted(grid["y_cc"], 0.0)
    lower_weight = (grid["y_cc"][upper] - 0.0)/(grid["y_cc"][upper] - grid["y_cc"][upper - 1])
    centerline = lower_weight*u[:, upper - 1] + (1.0 - lower_weight)*u[:, upper]

    # first sign change from reversed to forward flow in the fluid behind the body
    behind = (grid["x_cc"] > 0.5*DIAMETER) & (grid["ib_markers"][:, upper] == 0)
    x, velocity = grid["x_cc"][behind], centerline[behind]
    crossing = np.nonzero((velocity[:-1] < 0.0) & (velocity[1:] >= 0.0))[0][0]
    x_zero = x[crossing] - velocity[crossing]*(x[crossing + 1] - x[crossing])/(velocity[crossing + 1] - velocity[crossing])
    return (x_zero - 0.5*DIAMETER)/DIAMETER


def drag_coefficient(output_dir):
    forces = np.loadtxt(os.path.join(output_dir, "ib_forces.dat"))
    window = forces[-max(1, int(AVERAGE_WINDOW*len(forces))):, 2]
    return 2.0*window.mean()/(FREE_STREAM_VELOCITY**2*DIAMETER)


def observed_order(values, refinement_ratio=2.0):
    """Richardson order from the three finest grids."""
    coarse, medium, fine = values[-3:]
    return np.log(abs((coarse - medium)/(fine - medium)))/np.log(refinement_ratio)


def plot_results(results, resolutions, filepath):
    figure, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    for column, reynolds_number in enumerate(results):
        for row, (quantity, label) in enumerate((("drag_coefficient", "$C_D$"), ("wake_length", "$L/D$"))):
            axis = axes[row, column]
            axis.plot(resolutions, results[reynolds_number][quantity], "o-", color="tab:blue", label="present")
            axis.axhline(REFERENCE[reynolds_number][quantity], color="black", linestyle="--", label="Taira & Colonius")
            axis.set_xscale("log", base=2)
            axis.set_xticks(resolutions, [str(n) for n in resolutions])
            axis.set_ylabel(label)
            if row == 0:
                axis.set_title(f"Re = {reynolds_number}")
            if row == 1:
                axis.set_xlabel("$D/h$")
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(filepath, dpi=200)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reynolds_numbers", type=int, nargs="+", default=[20, 40])
    parser.add_argument("--resolutions", type=int, nargs="+", default=[16, 32, 64], help="cells per diameter")
    parser.add_argument("--t_stop", type=float, default=60.0)
    parser.add_argument("--poisson_solver", type=int, default=2, choices=[1, 2])
    parser.add_argument("--gpus", type=int, nargs="*", default=[], help="GPU ids used round robin, CPU when empty")
    parser.add_argument("--skip_runs", action="store_true", help="only post process existing runs")
    arguments = parser.parse_args()

    if not arguments.skip_runs:
        # largest runs first so each lands on its own GPU
        jobs = sorted(((reynolds_number, n) for reynolds_number in arguments.reynolds_numbers for n in arguments.resolutions),
                      key=lambda job: -job[1])
        gpus = [arguments.gpus[index % len(arguments.gpus)] if arguments.gpus else None for index in range(len(jobs))]
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            list(pool.map(lambda job, gpu: run_case(*job, arguments.poisson_solver, arguments.t_stop, gpu), jobs, gpus))

    results = {}
    for reynolds_number in arguments.reynolds_numbers:
        directories = [run_directory(reynolds_number, n, arguments.poisson_solver) for n in arguments.resolutions]
        results[reynolds_number] = {"drag_coefficient": [drag_coefficient(d) for d in directories],
                                    "wake_length": [wake_length(d) for d in directories]}
        for quantity, values in list(results[reynolds_number].items()):
            if len(values) >= 3:
                results[reynolds_number][f"{quantity}_order"] = observed_order(values)

    print(json.dumps({"resolutions": arguments.resolutions, "results": results}, indent=4))
    with open(os.path.join(CASE_DIR, "convergence", f"results_solver{arguments.poisson_solver}.json"), "w") as results_file:
        json.dump({"resolutions": arguments.resolutions, "reference": REFERENCE, "results": results}, results_file, indent=4)
    plot_results(results, arguments.resolutions, os.path.join(CASE_DIR, f"convergence_solver{arguments.poisson_solver}.png"))


if __name__ == "__main__":
    main()
