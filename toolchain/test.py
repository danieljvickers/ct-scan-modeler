"""Run the regression tests against golden files, or regenerate them, as ./mfc.sh test does for MFC."""

import glob
import os
import shutil
import subprocess
import time

import h5py
import numpy as np

from toolchain.build import build
from toolchain.case_validator import validate_case
from toolchain.computers import load_environment
from toolchain.paths import BUILD_ROOT, REPO_ROOT
from toolchain.run import INPUT_FILENAME, launch_command
from toolchain.test_cases import CASES

GOLDEN_ROOT = os.path.join(REPO_ROOT, "tests", "regression")
NUM_PROBES = 64             # field values sampled evenly through the flattened array
ABSOLUTE_FRACTION = 1e-9    # absolute tolerance as a fraction of the largest golden magnitude of each entry


def extract_values(output_dir):
    """Norms, sums, and probes of every field in the last save, plus the last body forces."""
    values = {}
    last_save = sorted(glob.glob(os.path.join(output_dir, "D", "fields_*.h5")))[-1]
    with h5py.File(last_save) as fields:
        for name in sorted(fields):
            data = fields[name][:].ravel()
            values[f"{name}_norms"] = [np.abs(data).mean(), np.sqrt((data**2).mean()), np.abs(data).max(), data.sum()]
            values[f"{name}_probes"] = list(data[np.linspace(0, data.size - 1, NUM_PROBES).astype(int)])
    force_path = os.path.join(output_dir, "D", "ib_forces.dat")
    if os.path.isfile(force_path):
        values["forces"] = list(np.loadtxt(force_path, ndmin=2)[-1, 2:])
    return values


def write_golden(path, test, values):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as golden:
        golden.write(f"# {test.name}\n")
        for key, numbers in values.items():
            golden.write(key + " " + " ".join(f"{float(number):.15e}" for number in numbers) + "\n")


def read_golden(path):
    values = {}
    with open(path) as golden:
        for line in golden:
            if line.startswith("#") or not line.strip():
                continue
            key, *numbers = line.split()
            values[key] = [float(number) for number in numbers]
    return values


def compare(values, golden, relative_tolerance):
    """Return a description of every entry outside |value - golden| <= rtol*|golden| + atol."""
    failures = []
    for key in sorted(set(values) | set(golden)):
        if key not in values or key not in golden or len(values[key]) != len(golden[key]):
            failures.append(f"{key}: present in only one of the run and the golden file, or a different length")
            continue
        expected, actual = np.array(golden[key]), np.array(values[key])
        tolerance = relative_tolerance*np.abs(expected) + ABSOLUTE_FRACTION*max(np.abs(expected).max(), 1e-300)
        for index in np.nonzero(~(np.abs(actual - expected) <= tolerance))[0]:
            failures.append(f"{key}[{index}]: {actual[index]:.15e} vs golden {expected[index]:.15e}")
    return failures


def run_case(test, executable, environment):
    """Run one test in build/tests/<id>, return its log path."""
    output_dir = os.path.join(BUILD_ROOT, "tests", test.id)
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)
    case = test.load()
    validate_case(case)
    input_path = os.path.join(output_dir, INPUT_FILENAME)
    with open(input_path, "w") as input_file:
        input_file.write(case.to_input_file(output_dir))
    log_path = os.path.join(output_dir, "log.txt")
    with open(log_path, "w") as log:
        result = subprocess.run(launch_command(test.num_gpus, None, environment) + [executable, input_path], cwd=output_dir,
                                env=environment, stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, f"solver exited with code {result.returncode}, see {log_path}"
    return output_dir


def test(arguments):
    selected = [case for case in CASES if not arguments.only or any(key == case.id or key in case.name for key in arguments.only)]
    assert selected, "ERROR :: no test matches --only"
    executable = build(None, arguments.computer, arguments.jobs, arguments.debug, arguments.verbose)
    environment = load_environment(arguments.computer)

    action = "generating" if arguments.generate else "running"
    print(f"INFO :: {action} {len(selected)} of {len(CASES)} regression tests")
    num_failed = 0
    for case in selected:
        start = time.perf_counter()
        golden_path = os.path.join(GOLDEN_ROOT, case.golden_id, "golden.txt")
        try:
            values = extract_values(run_case(case, executable, environment))
            if arguments.generate and case.golden_from is None:
                write_golden(golden_path, case, values)
                failures = []
            else:
                assert os.path.isfile(golden_path), f"no golden file {golden_path}, run ./ctm.sh test --generate"
                failures = compare(values, read_golden(golden_path), case.relative_tolerance)
        except AssertionError as error:
            failures = [str(error)]
        status = "FAIL" if failures else ("GENERATED" if arguments.generate and case.golden_from is None else "OK")
        print(f"{case.id} :: {status:<9} :: {time.perf_counter() - start:5.1f} s :: {case.name}")
        for failure in failures[:10]:
            print(f"    {failure}")
        if len(failures) > 10:
            print(f"    ... {len(failures) - 10} more")
        num_failed += bool(failures)

    assert num_failed == 0, f"ERROR :: {num_failed} of {len(selected)} tests failed"
    print(f"INFO :: all {len(selected)} tests passed" if not arguments.generate else "INFO :: golden files written to tests/regression")
