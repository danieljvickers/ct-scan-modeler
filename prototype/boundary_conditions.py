"""Domain boundary conditions as affine halo relations: halo = coefficient*interior + value."""

from dataclasses import dataclass

from toolchain import constants


@dataclass(frozen=True)
class Halo:
    coefficient: float
    value: float


def dirichlet(face_value):
    return Halo(-1.0, 2.0*face_value)  # linear through the face value


NEUMANN = Halo(1.0, 0.0)  # zero normal gradient
VARIABLES = ("u", "v", "p")


def _halos_for_side(boundary, normal_variable, vel_in):
    tangential_variable = "v" if normal_variable == "u" else "u"
    if boundary == constants.BC_DIRICHLET:
        return {"u": dirichlet(vel_in[0]), "v": dirichlet(vel_in[1]), "p": NEUMANN}
    if boundary == constants.BC_GHOST_EXTRAP:
        return {"u": NEUMANN, "v": NEUMANN, "p": dirichlet(0.0)}
    if boundary == constants.BC_SLIP_WALL:
        return {normal_variable: dirichlet(0.0), tangential_variable: NEUMANN, "p": NEUMANN}
    if boundary == constants.BC_NO_SLIP_WALL:
        return {"u": dirichlet(0.0), "v": dirichlet(0.0), "p": NEUMANN}
    raise ValueError(f"ERROR :: unsupported boundary condition {boundary}")


def get_halos(case):
    """Return halos[variable][(direction, side)] for u, v, and p on every domain side."""
    halos = {variable: {} for variable in VARIABLES}
    for direction, normal_variable in (("x", "u"), ("y", "v")):
        vel_in = [case[f"bc_{direction}%vel_in({k})"] for k in (1, 2)]
        for side in ("beg", "end"):
            side_halos = _halos_for_side(case[f"bc_{direction}%{side}"], normal_variable, vel_in)
            for variable in VARIABLES:
                halos[variable][(direction, side)] = side_halos[variable]
    return halos
