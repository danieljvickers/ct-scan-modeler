"""Reject case options the 2D prototype does not support, on top of the shared checks."""

from toolchain import constants
from toolchain.case_validator import validate_case


def check_case(case):
    validate_case(case)
    assert case["p"] == 0, "ERROR :: the prototype only supports 2D cases (p = 0)"
    boundaries = [case[f"bc_{direction}%{side}"] for direction in "xy" for side in ("beg", "end")]
    assert constants.BC_PERIODIC not in boundaries, "ERROR :: the prototype has no periodic boundaries, use the CUDA solver"
    for patch_id in range(1, case["num_patches"] + 1):
        assert case.family("patch_icpp", patch_id)["geometry"] == constants.ICPP_RECTANGLE, \
            "ERROR :: the prototype only has rectangle patches, use the CUDA solver"
