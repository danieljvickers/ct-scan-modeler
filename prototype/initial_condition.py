"""Initial velocity and pressure from rectangle patches, later patches overwrite earlier ones."""

import numpy as np

from prototype.backend import to_device


def initial_condition(case, grid):
    x_cc, y_cc = grid.cell_centers()
    fields = {name: np.zeros(grid.shape) for name in ("u", "v", "p")}
    for patch_id in range(1, case["num_patches"] + 1):
        patch = case.family("patch_icpp", patch_id)
        inside = ((np.abs(x_cc - patch["x_centroid"]) <= 0.5*patch["length_x"]) &
                  (np.abs(y_cc - patch["y_centroid"]) <= 0.5*patch["length_y"]))
        fields["u"][inside] = patch["vel(1)"]
        fields["v"][inside] = patch["vel(2)"]
        fields["p"][inside] = patch["pres"]
    velocities = {variable: to_device(fields[variable].ravel()) for variable in ("u", "v")}
    return velocities, to_device(fields["p"].ravel())
