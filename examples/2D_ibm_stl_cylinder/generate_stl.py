#!/usr/bin/env python3
"""Write a unit diameter disk centered at the origin as a flat ASCII STL fan."""

import argparse

# math and scientific libraries
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_segments", type=int, default=512, help="polygon segments around the circumference")
parser.add_argument("--output", default="cylinder.stl")
arguments = parser.parse_args()

radius = 0.5
angles = np.linspace(0.0, 2.0*np.pi, arguments.num_segments + 1)
rim = np.column_stack([radius*np.cos(angles), radius*np.sin(angles)])

with open(arguments.output, "w") as stl_file:
    stl_file.write("solid cylinder\n")
    for start, end in zip(rim[:-1], rim[1:]):
        stl_file.write("  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n")
        stl_file.write(f"      vertex {start[0]:.12e} {start[1]:.12e} 0\n      vertex {end[0]:.12e} {end[1]:.12e} 0\n")
        stl_file.write("    endloop\n  endfacet\n")
    stl_file.write("endsolid cylinder\n")
