"""Run a case.py, validate its parameters, and fill in defaults."""

import json
import os
import subprocess
import sys

from toolchain import parameters


def _convert(name, value, kind):
    if kind is bool:
        if isinstance(value, bool):
            return value
        assert value in ("T", "F"), f"ERROR :: {name} must be 'T' or 'F', got {value!r}"
        return value == "T"
    if kind is int:
        assert float(value) == int(value), f"ERROR :: {name} must be an integer, got {value!r}"
        return int(value)
    return kind(value)


class Case:
    """Validated case parameters with MFC style keys."""

    def __init__(self, raw_parameters, case_dir):
        self.values = {}
        for name, value in raw_parameters.items():
            definition = parameters.lookup(name)
            assert definition is not None, f"ERROR :: unknown case parameter {name}"
            self.values[name] = _convert(name, value, definition.kind)
        self.values["case_dir"] = os.path.abspath(case_dir)

        for name, definition in parameters.SCALARS.items():
            if name not in self.values:
                assert definition.default is not None, f"ERROR :: required case parameter {name} is missing"
                self.values[name] = definition.default

    def __getitem__(self, name):
        return self.values[name]

    def family(self, base_name, index):
        """Return every attribute of one indexed entry, e.g. family("patch_ib", 1)["geometry"]."""
        _, attributes = parameters.FAMILIES[base_name]
        entry = {}
        for attribute, definition in attributes.items():
            name = f"{base_name}({index})%{attribute}"
            value = self.values.get(name, definition.default)
            assert value is not None, f"ERROR :: required case parameter {name} is missing"
            entry[attribute] = value
        return entry

    def to_json(self):
        return json.dumps(self.values, indent=4)

    def to_input_file(self):
        """Flat 'key value' lines for the CUDA solver: every scalar plus the patches in use, with STL paths made absolute."""
        def formatted(value):
            if isinstance(value, bool):
                return "T" if value else "F"
            return repr(value) if isinstance(value, float) else str(value)

        lines = [f"{name} {formatted(value)}" for name, value in sorted(self.values.items()) if name in parameters.SCALARS]
        for base_name, count_name in (("patch_icpp", "num_patches"), ("patch_ib", "num_ibs")):
            for index in range(1, self[count_name] + 1):
                for attribute, value in self.family(base_name, index).items():
                    if base_name == "patch_ib" and attribute == "geometry":
                        value = os.path.join(self["case_dir"], value)
                    lines.append(f"{base_name}({index})%{attribute} {formatted(value)}")
        return "\n".join(lines) + "\n"


def load_case(case_path):
    """Execute case.py and return the validated Case; case files take no command line arguments."""
    case_path = os.path.abspath(case_path)
    assert os.path.isfile(case_path), f"ERROR :: case file {case_path} not found"
    result = subprocess.run([sys.executable, case_path], capture_output=True, text=True,
                            cwd=os.path.dirname(case_path))
    assert result.returncode == 0, f"ERROR :: {case_path} failed\n{result.stderr}"
    return Case(json.loads(result.stdout), os.path.dirname(case_path))
