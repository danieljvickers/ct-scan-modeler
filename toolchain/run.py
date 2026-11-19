"""Validate a case, build the solver for it, and launch one MPI rank per GPU."""

import os
import shutil
import subprocess

from toolchain.build import build
from toolchain.case import load_case
from toolchain.case_validator import validate_case
from toolchain.computers import load_environment

INPUT_FILENAME = "simulation.json"


def launch_command(num_gpus, hostfile, environment):
    """srun inside a Slurm allocation, mpirun otherwise."""
    path = environment.get("PATH")
    if "SLURM_JOB_ID" in environment and shutil.which("srun", path=path):
        return ["srun", "--ntasks", str(num_gpus)]
    assert shutil.which("mpirun", path=path), "ERROR :: mpirun not found, load an MPI environment with -c"
    command = ["mpirun", "-np", str(num_gpus)]
    return command + ["--hostfile", hostfile] if hostfile else command


def run(arguments, case_arguments):
    assert arguments.num_gpus >= 1, "ERROR :: -n must be at least 1"
    case = load_case(arguments.case_path, case_arguments)
    validate_case(case)

    executable = build(None if arguments.no_case_optimization else case, arguments.computer, arguments.jobs, arguments.debug,
                       arguments.verbose)

    # write the validated, default filled case next to case.py for the solver to read
    input_path = os.path.join(case["case_dir"], INPUT_FILENAME)
    with open(input_path, "w") as input_file:
        input_file.write(case.to_json())

    environment = load_environment(arguments.computer)
    command = launch_command(arguments.num_gpus, arguments.hostfile, environment) + [executable, input_path]
    print(f"INFO :: running {' '.join(command)}")
    subprocess.run(command, cwd=case["case_dir"], env=environment, check=True)
