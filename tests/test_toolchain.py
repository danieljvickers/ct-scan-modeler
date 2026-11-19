"""Unit tests for the toolchain: computers, case headers, and build configurations."""

import os

from toolchain import parameters
from toolchain.build import build_configuration, case_header
from toolchain.case import load_case
from toolchain.computers import computer_name, list_computers, load_environment

CYLINDER_CASE = os.path.join(os.path.dirname(__file__), "..", "examples", "2D_ibm_stl_cylinder", "case.py")


def test_wingtip_alias_and_environment():
    assert list_computers()["w"] == list_computers()["wingtip"]
    assert computer_name("w") == "wingtip" and computer_name(None) == "local"
    assert load_environment("w")["CTM_CUDA_ARCHITECTURES"] == "80"


def test_case_header_holds_every_case_optimized_parameter():
    header = case_header(load_case(CYLINDER_CASE))
    assert "constexpr int m = 1919;" in header and "constexpr bool viscous = true;" in header
    for name in parameters.CASE_OPTIMIZATION:
        assert f" {name} = " in header


def test_build_configuration_changes_only_with_case_optimized_parameters():
    reference, longer, finer = (load_case(CYLINDER_CASE) for _ in range(3))
    longer.values["t_step_stop"] *= 2
    finer.values["m"] = 2*reference["m"] + 1
    assert build_configuration(reference, "w", False) == build_configuration(longer, "w", False)
    assert build_configuration(reference, "w", False) != build_configuration(finer, "w", False)
    assert build_configuration(None, "w", False)["case_optimization"] is None


def test_regression_compare_accepts_round_off_and_catches_changes():
    from toolchain.test import compare
    golden = {"u_norms": [1.0, 2.0, 0.5, 1.0e-15], "forces": [0.77, 1.0e-16]}
    round_off = {"u_norms": [1.0 + 1e-12, 2.0, 0.5, -3.0e-15], "forces": [0.77, -2.0e-16]}
    changed = {"u_norms": [1.0 + 1e-6, 2.0, 0.5, 1.0e-15], "forces": [0.77, 1.0e-16]}
    assert compare(round_off, golden, 1e-8) == []
    assert len(compare(changed, golden, 1e-8)) == 1
    assert compare({"u_norms": golden["u_norms"]}, golden, 1e-8)  # a missing entry fails
