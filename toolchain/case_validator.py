"""Case checks shared by the Python prototype and the CUDA solver."""

from toolchain import constants


def num_dims(case):
    return 1 + (case["n"] > 0) + (case["p"] > 0)


def validate_case(case):
    dims = num_dims(case)
    assert dims >= 2, "ERROR :: cases must be 2D (n > 0) or 3D (n > 0, p > 0)"
    assert case["num_fluids"] == 1, "ERROR :: only num_fluids = 1 is supported"
    assert case["time_stepper"] == constants.TIME_STEPPER_AB2_CN, "ERROR :: only time_stepper = 1 (AB2/CN) is supported"
    assert case["poisson_solver"] in (constants.POISSON_SPARSE_LU, constants.POISSON_SPECTRAL_KRYLOV), \
        "ERROR :: poisson_solver must be 1 (sparse LU) or 2 (spectral Krylov)"
    if case["poisson_solver"] == constants.POISSON_SPECTRAL_KRYLOV:
        assert not any(case[f"stretch_{d}"] for d in "xyz"), "ERROR :: poisson_solver = 2 needs a uniform grid, turn off stretching"
        assert 0 < case["krylov_tolerance"] < 1 and case["krylov_restart"] > 0, "ERROR :: invalid Krylov settings"
    assert case["dt"] > 0, "ERROR :: dt must be positive"
    assert case["t_step_save"] > 0, "ERROR :: t_step_save must be positive"
    if case["viscous"]:
        assert case["fluid_pp(1)%Re(1)"] > 0, "ERROR :: viscous cases need fluid_pp(1)%Re(1) > 0"

    boundaries = [case[f"bc_{direction}%{side}"] for direction in "xyz"[:dims] for side in ("beg", "end")]
    for boundary in boundaries:
        assert boundary in constants.SUPPORTED_BCS, f"ERROR :: boundary condition {boundary} is not supported"
    assert constants.BC_GHOST_EXTRAP in boundaries, "ERROR :: one boundary must be -3 (outflow) to fix the pressure level"

    for patch_id in range(1, case["num_patches"] + 1):
        assert case.family("patch_icpp", patch_id)["geometry"] == constants.ICPP_RECTANGLE, \
            f"ERROR :: patch_icpp({patch_id}) must be a rectangle (geometry = 3)"

    if not case["ib"]:
        return
    assert case["ibm_ip_interpolation"] == constants.IP_INTERP_BILINEAR, "ERROR :: only ibm_ip_interpolation = 2 is supported"
    assert case["gp_layers"] >= 1, "ERROR :: gp_layers must be at least 1"
    stl_geometry = constants.IB_STL_2D if dims == 2 else constants.IB_STL_3D
    for patch_id in range(1, case["num_ibs"] + 1):
        patch = case.family("patch_ib", patch_id)
        assert patch["geometry"] == stl_geometry, f"ERROR :: patch_ib({patch_id}) must be a {dims}D STL (geometry = {stl_geometry})"
        assert not patch["slip"], f"ERROR :: patch_ib({patch_id}) slip walls are not supported"
        assert 1 <= patch["model_id"] <= case["num_stl_models"], f"ERROR :: patch_ib({patch_id})%model_id is out of range"
