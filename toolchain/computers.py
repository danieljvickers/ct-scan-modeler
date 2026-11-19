"""Computer environments: one shell script per machine in toolchain/computers, selected by name or alias."""

import os
import subprocess

COMPUTERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "computers")


def _aliases(script_path):
    with open(script_path) as script:
        for line in script:
            if line.startswith("# aliases:"):
                return line.split(":", 1)[1].split()
    return []


def list_computers():
    """Map every name and alias to its environment script."""
    computers = {}
    for filename in sorted(os.listdir(COMPUTERS_DIR)):
        if not filename.endswith(".sh"):
            continue
        script_path = os.path.join(COMPUTERS_DIR, filename)
        for name in [filename[:-3], *_aliases(script_path)]:
            computers[name] = script_path
    return computers


def computer_name(computer):
    """Canonical computer name, the script filename without .sh, or 'local' for the current environment."""
    return "local" if computer is None else os.path.basename(list_computers()[computer])[:-3]


def load_environment(computer):
    """Return the process environment after sourcing the computer script, or the current one when computer is None."""
    if computer is None:
        return dict(os.environ)
    computers = list_computers()
    assert computer in computers, f"ERROR :: unknown computer {computer}, choose from {', '.join(sorted(computers))}"
    result = subprocess.run(["bash", "-c", f"source '{computers[computer]}' > /dev/null && env -0"], capture_output=True, check=True)
    return dict(entry.split("=", 1) for entry in result.stdout.decode().split("\0") if "=" in entry)
