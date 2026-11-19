"""Uniform or tanh-stretched rectilinear grids, following MFC's m_grid."""

import numpy as np


def _cell_boundaries(case, direction, num_cells):
    beg, end = case[f"{direction}_domain%beg"], case[f"{direction}_domain%end"]
    boundaries = np.linspace(beg, end, num_cells + 1)
    if not case[f"stretch_{direction}"]:
        return boundaries

    # hyperbolic tangent stretching that keeps [x_a, x_b] near uniform, same map as MFC
    length = abs(end - beg)
    rate = case[f"a_{direction}"]
    cluster_beg, cluster_end = case[f"{direction}_a"]/length, case[f"{direction}_b"]/length
    log_cosh = lambda z: np.logaddexp(z, -z) - np.log(2.0)  # log(cosh(z)) without overflow
    boundaries = boundaries/length
    for _ in range(case[f"loops_{direction}"]):
        boundaries = boundaries/rate*(rate + log_cosh(rate*(boundaries - cluster_beg))
                                      + log_cosh(rate*(boundaries - cluster_end))
                                      - 2.0*log_cosh(rate*(cluster_end - cluster_beg)/2.0))
    return boundaries*length


class Grid:
    """Cell boundaries (x_cb), centers (x_cc), and widths (dx) in each direction."""

    def __init__(self, case):
        self.m, self.n = case["m"], case["n"]
        self.shape = (self.m + 1, self.n + 1)
        self.num_cells = self.shape[0]*self.shape[1]

        self.x_cb = _cell_boundaries(case, "x", self.m + 1)
        self.y_cb = _cell_boundaries(case, "y", self.n + 1)
        self.x_cc = 0.5*(self.x_cb[1:] + self.x_cb[:-1])
        self.y_cc = 0.5*(self.y_cb[1:] + self.y_cb[:-1])
        self.dx = np.diff(self.x_cb)
        self.dy = np.diff(self.y_cb)

        self.cell_volumes = np.outer(self.dx, self.dy)
        self.min_cell_width = min(self.dx.min(), self.dy.min())

    def cell_centers(self):
        """Return the (x, y) cell center coordinates as two arrays of the grid shape."""
        return np.meshgrid(self.x_cc, self.y_cc, indexing="ij")
