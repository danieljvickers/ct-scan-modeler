"""Run a command behind a transient progress bar and print its captured output only if it fails, as MFC does."""

import re
import subprocess

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

MAKE_PROGRESS = re.compile(r"^\[\s*(\d+)%\]")      # [ 42%] Building CUDA object ...
NINJA_PROGRESS = re.compile(r"^\[(\d+)/(\d+)\]")  # [3/7] Building CUDA object ...

console = Console()


def run_with_progress(command, label, environment, verbose=False):
    """Show one bar while the command runs, then clear it; on failure print everything the command wrote."""
    if verbose:
        subprocess.run(command, env=environment, check=True)
        return

    output = []
    columns = (SpinnerColumn(), TextColumn(f"[bold blue]{label}"), BarColumn(bar_width=30), TaskProgressColumn(), TimeElapsedColumn())
    with Progress(*columns, console=console, transient=True, disable=not console.is_terminal) as progress:  # no bar in logs or pipes
        task = progress.add_task(label, total=None)  # no total pulses the bar until make reports a percentage
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            output.append(line)
            make_match, ninja_match = MAKE_PROGRESS.match(line), NINJA_PROGRESS.match(line)
            if make_match:
                progress.update(task, total=100, completed=int(make_match.group(1)))
            elif ninja_match:
                progress.update(task, total=int(ninja_match.group(2)), completed=int(ninja_match.group(1)))
        process.wait()

    if process.returncode != 0:
        console.print(Panel(Text("".join(output).strip()), title=f"{label} failed", border_style="red"))
        raise subprocess.CalledProcessError(process.returncode, command)
