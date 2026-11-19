"""Field and immersed boundary force output, written to the case D directory as MFC does."""

import os

import numpy as np

from prototype.backend import to_host, xp


class DataOutput:

    def __init__(self, case, grid, immersed_boundaries, output_dir):
        self.case = case
        self.grid = grid
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "case.json"), "w") as case_file:
            case_file.write(case.to_json())

        markers = np.zeros(grid.shape, dtype=np.int32) if immersed_boundaries is None else immersed_boundaries.ib_markers
        np.savez(os.path.join(output_dir, "grid.npz"), x_cb=grid.x_cb, y_cb=grid.y_cb, x_cc=grid.x_cc, y_cc=grid.y_cc, ib_markers=markers)

        self.force_file = None
        self.force_buffer = []  # device force arrays, copied to the host in one batch at each save
        if case["ib"] and case["ib_force_wrt"]:
            self.force_file = open(os.path.join(output_dir, "ib_forces.dat"), "w")
            columns = " ".join(f"force_x({i}) force_y({i})" for i in range(1, case["num_ibs"] + 1))
            self.force_file.write(f"# step time {columns}\n")

    def write_fields(self, step, velocities, pressure):
        if not self.case["prim_vars_wrt"]:
            return
        np.savez(os.path.join(self.output_dir, f"fields_{step:08d}.npz"), step=step, time=step*self.case["dt"],
                 u=to_host(velocities["u"]).reshape(self.grid.shape), v=to_host(velocities["v"]).reshape(self.grid.shape),
                 p=to_host(pressure).reshape(self.grid.shape))

    def write_forces(self, step, forces):
        if self.force_file is not None:
            self.force_buffer.append((step, forces))

    def flush_forces(self):
        if not self.force_buffer:
            return
        steps = [step for step, _ in self.force_buffer]
        values = to_host(xp.stack([forces.ravel() for _, forces in self.force_buffer]))
        for step, row in zip(steps, values):
            self.force_file.write(f"{step} {step*self.case['dt']:.10e} " + " ".join(f"{value:.10e}" for value in row) + "\n")
        self.force_file.flush()
        self.force_buffer = []

    def close(self):
        if self.force_file is not None:
            self.flush_forces()
            self.force_file.close()
