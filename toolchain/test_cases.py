"""Regression tests built from the examples at low resolution, in the spirit of MFC's toolchain/mfc/test/cases.py."""

import hashlib
import os
from dataclasses import dataclass, field

from toolchain.case import load_case
from toolchain.paths import REPO_ROOT

NUM_STEPS = 20  # every test runs this many steps and compares the last save


@dataclass
class TestCase:
    name: str
    example: str                      # directory under examples/
    coarsening: int = 4               # cells per direction are divided by this, dt grows by it to keep the CFL number
    overrides: dict = field(default_factory=dict)
    num_gpus: int = 1
    golden_from: str = None           # name of the test whose golden file this one must reproduce
    relative_tolerance: float = 1e-8

    @property
    def id(self):
        return hashlib.sha256(self.name.encode()).hexdigest()[:8].upper()

    @property
    def golden_id(self):
        return self.id if self.golden_from is None else TestCase(self.golden_from, self.example).id

    def load(self):
        """The example case, coarsened and shortened, with this test's overrides applied."""
        case = load_case(os.path.join(REPO_ROOT, "examples", self.example, "case.py"))
        for name in ("m", "n", "p"):
            if case[name] == 0:
                continue
            assert (case[name] + 1) % self.coarsening == 0, f"ERROR :: {name} + 1 of {self.example} is not divisible by {self.coarsening}"
            case.values[name] = (case[name] + 1)//self.coarsening - 1
        case.values.update({"dt": case["dt"]*self.coarsening, "t_step_start": 0, "t_step_stop": NUM_STEPS, "t_step_save": NUM_STEPS,
                            "run_time_info": False})
        case.values.update(self.overrides)
        return case


CYLINDER = "2D_ibm_stl_cylinder"

CASES = [
    TestCase("2D cylinder", CYLINDER),
    TestCase("2D cylinder on 2 GPUs", CYLINDER, num_gpus=2, golden_from="2D cylinder"),
    TestCase("2D cylinder without immersed boundary", CYLINDER, overrides={"ib": False, "num_ibs": 0}),
    TestCase("2D cylinder with no-slip walls", CYLINDER, overrides={"bc_y%beg": -16, "bc_y%end": -16}),
    TestCase("2D cylinder with rk3", CYLINDER, overrides={"rk_order": 3}),
    TestCase("2D cylinder with rk3 and a single projection", CYLINDER, overrides={"rk_order": 3, "rk_single_projection": True}),
]
