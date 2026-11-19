"""Third party libraries built once per computer into build/dependencies/<computer>/install, currently parallel HDF5."""

import os
import shutil

from toolchain.computers import computer_name
from toolchain.paths import BUILD_ROOT, REPO_ROOT
from toolchain.progress import run_with_progress

DEPENDENCIES_SOURCE = os.path.join(REPO_ROOT, "toolchain", "dependencies")
DEPENDENCIES_VERSION = "hdf5-1.14.6"  # bump to force a rebuild when toolchain/dependencies/CMakeLists.txt changes


def ensure_dependencies(computer, environment, jobs, verbose):
    """Build the dependencies if this computer does not have them yet and return the install prefix."""
    root = os.path.join(BUILD_ROOT, "dependencies", computer_name(computer))
    install_dir = os.path.join(root, "install")
    stamp = os.path.join(install_dir, "ctm_dependencies.txt")
    if os.path.isfile(stamp) and open(stamp).read().strip() == DEPENDENCIES_VERSION:
        return install_dir

    mpicc = shutil.which("mpicc", path=environment.get("PATH"))
    assert mpicc, "ERROR :: mpicc not found, load an MPI environment with -c"
    dependency_environment = dict(environment, OMPI_CC="gcc")  # HDF5 builds with gcc under the MPI wrapper
    staging_dir = os.path.join(root, "staging")
    print(f"INFO :: building {DEPENDENCIES_VERSION} for {computer_name(computer)} once, this takes a few minutes")
    run_with_progress(["cmake", "-S", DEPENDENCIES_SOURCE, "-B", staging_dir, f"-DCMAKE_INSTALL_PREFIX={install_dir}",
                       f"-DCMAKE_C_COMPILER={mpicc}"], "Configuring dependencies", dependency_environment, verbose)
    run_with_progress(["cmake", "--build", staging_dir, "--parallel", str(jobs)], "Building dependencies", dependency_environment, verbose)
    with open(stamp, "w") as stamp_file:
        stamp_file.write(DEPENDENCIES_VERSION + "\n")
    return install_dir
