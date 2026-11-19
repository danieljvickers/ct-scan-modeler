"""Reject case options the 2D prototype does not support, on top of the shared checks."""

from toolchain.case_validator import validate_case


def check_case(case):
    validate_case(case)
    assert case["p"] == 0, "ERROR :: the prototype only supports 2D cases (p = 0)"
