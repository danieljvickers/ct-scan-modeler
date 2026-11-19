"""Mark cells inside immersed boundary patches, following MFC's m_ib_patches."""

import numpy as np

from prototype.model import Model2D


def load_models(case):
    """One model per immersed boundary patch."""
    return [Model2D(case.family("patch_ib", patch_id), case["case_dir"]) for patch_id in range(1, case["num_ibs"] + 1)]


def compute_ib_markers(case, grid, models):
    """Return an integer array of the grid shape: 0 in fluid, the patch id inside a body."""
    ib_markers = np.zeros(grid.shape, dtype=np.int32)
    x_cc, y_cc = grid.cell_centers()
    for patch_id in range(1, case["num_ibs"] + 1):
        model = models[patch_id - 1]

        # only test cells inside the model bounding box
        candidates = ((x_cc >= model.bounding_box[0, 0]) & (x_cc <= model.bounding_box[1, 0]) &
                      (y_cc >= model.bounding_box[0, 1]) & (y_cc <= model.bounding_box[1, 1]))
        points = np.column_stack([x_cc[candidates], y_cc[candidates]])
        inside = np.zeros(grid.shape, dtype=bool)
        inside[candidates] = model.is_inside(points)
        ib_markers[inside] = patch_id
    return ib_markers
