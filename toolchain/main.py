"""Command line entry point behind ./ctm.sh <run|build|test|model>."""

import argparse
import os
import subprocess
import sys

from toolchain.build import build
from toolchain.computers import list_computers
from toolchain.run import run


def add_build_arguments(parser):
    parser.add_argument("-c", "--computer", choices=sorted(list_computers()), default=None,
                        help="computer environment from toolchain/computers, defaults to the current shell")
    parser.add_argument("-j", "--jobs", type=int, default=min(16, os.cpu_count() or 1), help="parallel build jobs")
    parser.add_argument("--debug", action="store_true", help="debug build instead of release")
    parser.add_argument("-v", "--verbose", action="store_true", help="print the full CMake and compiler output")


def make_parser():
    parser = argparse.ArgumentParser(prog="./ctm.sh", description="CT scan modeler toolchain")
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", allow_abbrev=False, help="build case optimized and run a case, arguments after the case go to case.py")
    run_parser.add_argument("case_path", help="path to case.py")
    run_parser.add_argument("-n", "--num_gpus", type=int, default=1, help="number of GPUs, one MPI rank each")
    run_parser.add_argument("--hostfile", default=None, help="MPI hostfile for multinode runs outside Slurm")
    run_parser.add_argument("--no-case-optimization", dest="no_case_optimization", action="store_true",
                            help="use the generic build instead of compiling the case parameters in")
    add_build_arguments(run_parser)

    build_parser = commands.add_parser("build", allow_abbrev=False, help="generic build without case optimization")
    add_build_arguments(build_parser)

    commands.add_parser("test", help="not implemented yet")
    commands.add_parser("model", help="not implemented yet")
    return parser


def main():
    arguments, extra_arguments = make_parser().parse_known_args()
    if extra_arguments and arguments.command != "run":
        sys.exit(f"ERROR :: unrecognized arguments: {' '.join(extra_arguments)}")
    try:
        if arguments.command == "run":
            run(arguments, extra_arguments)
        elif arguments.command == "build":
            print(f"INFO :: installed {build(None, arguments.computer, arguments.jobs, arguments.debug, arguments.verbose)}")
        else:
            sys.exit(f"ERROR :: ./ctm.sh {arguments.command} is not implemented yet")
    except AssertionError as error:
        sys.exit(str(error))
    except subprocess.CalledProcessError as error:
        sys.exit(f"ERROR :: command failed with exit code {error.returncode}: {' '.join(map(str, error.cmd))}")


if __name__ == "__main__":
    main()
