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
    header = case_header(load_case(CYLINDER_CASE, ["--cells_per_diameter", "8"]))
    assert "constexpr int m = 479;" in header and "constexpr bool viscous = true;" in header
    for name in parameters.CASE_OPTIMIZATION:
        assert f" {name} = " in header


def test_build_configuration_changes_only_with_case_optimized_parameters():
    coarse = load_case(CYLINDER_CASE, ["--cells_per_diameter", "8"])
    longer = load_case(CYLINDER_CASE, ["--cells_per_diameter", "8", "--t_stop", "5"])
    finer = load_case(CYLINDER_CASE, ["--cells_per_diameter", "16"])
    assert build_configuration(coarse, "w", False) == build_configuration(longer, "w", False)
    assert build_configuration(coarse, "w", False) != build_configuration(finer, "w", False)
    assert build_configuration(None, "w", False)["case_optimization"] is None
