"""Grow a tumor next to the airway and deform the airway STL around it.

Growth follows the Gompertz law, calibrated to lung cancer by the CT volume doubling time of each histology at detection
size and a carrying capacity at the lethal tumor burden. The tumor pushes the tissue around it straight outward with the
volume conserving map r^3 = r0^3 + R^3, applied to the STL vertices. The push fades to zero far from the tumor and next to
anchored points, such as the straight inlet and outlet extensions, so those stay exactly where they are.
"""
import numpy as np
from scipy.spatial import cKDTree

from . import mesh, tree

# pooled mean CT volume doubling time of solid lung cancer, Jiang et al., Eur. J. Cancer 212 (2024) 114339
DOUBLING_TIME_DAYS = {"adenocarcinoma": 223.0, "squamous cell": 140.0, "small cell": 73.0}
DETECTION_VOLUME_CM3 = 1.0     # about 1e9 cells, near the smallest tumor CT finds, where the doubling times are measured
CARRYING_CAPACITY_CM3 = 1.0e3  # about 1e12 cells, the lethal burden used for Gompertz growth by Norton et al., Nature 264 (1976)
DAYS_PER_MONTH = 365.25/12
PUSH_FULL_MM = 30.0            # the push is exact out to here from the tumor center
PUSH_ZERO_MM = 45.0            # and fades smoothly to nothing here
ANCHOR_BLEND_MM = 10.0         # the push also fades to nothing over this distance next to anchored points
MAX_EDGE_MM = 0.6              # triangles stretched past this beside the tumor are split, about the size of the CT voxels
STRETCH_LIMIT = 1.2            # only edges the push lengthens by more than this are split


def gompertz_volume(days, doubling_time_days, initial_volume=DETECTION_VOLUME_CM3, capacity=CARRYING_CAPACITY_CM3):
    """Tumor volume in cm^3 after the given days, starting at initial_volume and doubling in doubling_time_days at that size."""
    decay_rate = np.log(2.0)/(doubling_time_days*np.log(capacity/initial_volume))
    return capacity*(initial_volume/capacity)**np.exp(-decay_rate*np.asarray(days, dtype=float))


def sphere_radius_mm(volume_cm3):
    return np.cbrt(3.0*np.asarray(volume_cm3)*1e3/(4.0*np.pi))


def _smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value*value*(3.0 - 2.0*value)


class Push:
    """The tissue push of one tumor of a given radius. Points are in mm.

    Every point moves along its ray from the tumor center, by the volume conserving amount times a weight that is one near the
    tumor and falls to zero far from it and next to the anchors.
    """

    def __init__(self, center_mm, radius_mm, anchors_mm=None, full_mm=PUSH_FULL_MM, zero_mm=PUSH_ZERO_MM,
                 anchor_blend_mm=ANCHOR_BLEND_MM):
        self.center = np.asarray(center_mm, dtype=float)
        self.radius = float(radius_mm)
        self.full_mm, self.zero_mm, self.anchor_blend_mm = full_mm, zero_mm, anchor_blend_mm
        self.anchors = cKDTree(anchors_mm) if anchors_mm is not None and len(anchors_mm) else None

    def weight(self, points):
        weight = _smoothstep((self.zero_mm - np.linalg.norm(points - self.center, axis=-1))/(self.zero_mm - self.full_mm))
        if self.anchors is not None:
            weight = weight*_smoothstep(self.anchors.query(points.reshape(-1, 3))[0].reshape(weight.shape)/self.anchor_blend_mm)
        return weight

    def _distance_along_ray(self, start, direction):
        point = self.center + start[..., None]*direction
        return start + self.weight(point)*(np.cbrt(start**3 + self.radius**3) - start)

    def __call__(self, points):
        offset = points - self.center
        start = np.linalg.norm(offset, axis=-1)
        direction = np.divide(offset, start[..., None], out=np.zeros_like(offset), where=start[..., None] > 0)
        return self.center + self._distance_along_ray(start, direction)[..., None]*direction

    def pull_back(self, points):
        """Where the tissue now at each point started, NaN inside the tumor. Solved along each ray by bisection."""
        offset = points - self.center
        distance = np.linalg.norm(offset, axis=-1)
        direction = np.divide(offset, distance[..., None], out=np.zeros_like(offset), where=distance[..., None] > 0)
        low, high = np.zeros_like(distance), distance.copy()
        for _ in range(50):
            middle = 0.5*(low + high)
            beyond = self._distance_along_ray(middle, direction) > distance
            high, low = np.where(beyond, middle, high), np.where(beyond, low, middle)
        start = 0.5*(low + high)
        start[self._distance_along_ray(np.zeros_like(distance), direction) > distance] = np.nan  # no tissue reaches here
        return self.center + start[..., None]*direction


def extension_points(vertices_mm, spacing_mm, directions, match_fraction=0.9):
    """STL vertices on the straight extensions that tree.cut_and_extrude sweeps out of the domain, e.g. directions ["+z", "-x"].

    An extension repeats the same cross section at every voxel layer, so its wall vertices sit at the same transverse position
    layer after layer. Walking in from the end of the surface, the extension lasts as long as a layer keeps those positions.
    """
    vertices_mm = np.asarray(vertices_mm, dtype=float)
    frozen = np.zeros(len(vertices_mm), dtype=bool)
    for name in set(directions):
        axis, sign = tree.AXES.get(name, (2, 1) if name == "+z" else None)
        step = spacing_mm[axis]
        transverse = [a for a in range(3) if a != axis]
        keys = np.round(vertices_mm[:, transverse]*1e3).astype(np.int64)
        depth = (vertices_mm[:, axis].max() - vertices_mm[:, axis]) if sign > 0 else (vertices_mm[:, axis] - vertices_mm[:, axis].min())
        # wall vertices lie on voxel layers, whose offset from the rounded end cap is the most common fractional depth near the end
        fractions = np.round((depth[depth < 20*step]/step) % 1.0, 3)
        values, counts = np.unique(fractions, return_counts=True)
        depth = depth - values[np.argmax(counts)]*step
        layer = np.round(depth/step).astype(int)
        on_layer = np.abs(depth - layer*step) < 1e-3
        # the reference layer sits a few voxels in from the closing cap, past where the cap smoothing reaches
        reference = {tuple(k) for k in keys[on_layer & (layer == 4)]}
        length = 4
        while True:
            here = keys[on_layer & (layer == length + 1)]
            if len(here) == 0 or np.mean([tuple(k) in reference for k in here]) < match_fraction:
                break
            length += 1
        inside = np.array([tuple(k) in reference for k in keys])
        frozen |= inside & (layer <= length)
        frozen |= depth < 4*step  # the cap itself
    return vertices_mm[frozen]


def wall_point(vertices, faces, inside_mm, direction):
    """First crossing of the airway wall from a point inside the airway along a direction (Moller-Trumbore on every face)."""
    direction = np.asarray(direction, dtype=float)/np.linalg.norm(direction)
    corner, edge_1, edge_2 = vertices[faces[:, 0]], vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]
    normal = np.cross(direction, edge_2)
    determinant = np.einsum("ij,ij->i", edge_1, normal)
    usable = np.abs(determinant) > 1e-12
    inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=usable)
    to_origin = inside_mm - corner
    u = np.einsum("ij,ij->i", to_origin, normal)*inverse
    cross = np.cross(to_origin, edge_1)
    v = (cross @ direction)*inverse
    distance = np.einsum("ij,ij->i", edge_2, cross)*inverse
    hit = usable & (u >= 0) & (v >= 0) & (u + v <= 1) & (distance > 0)
    if not hit.any():
        raise ValueError("the ray never leaves the airway")
    return np.asarray(inside_mm, dtype=float) + distance[hit].min()*direction


def refine_for_push(vertices, faces, push, max_edge_mm=MAX_EDGE_MM):
    """Split edges that the push would stretch past max_edge_mm, keeping the surface watertight.

    Every marked edge gets one midpoint shared by both triangles on it, and each triangle is split by its marked edges.
    """
    vertices, faces = vertices.copy(), faces.copy()
    for _ in range(12):
        pushed = push(vertices)
        edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        stretched = np.linalg.norm(pushed[edges[:, 0]] - pushed[edges[:, 1]], axis=1)
        original = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
        long = (stretched > max_edge_mm) & (stretched > STRETCH_LIMIT*original)  # edges the push barely stretches stay as they are
        if not long.any():
            break
        unique_edges, inverse = np.unique(edges[long], axis=0, return_inverse=True)
        midpoints = len(vertices) + np.arange(len(unique_edges))
        vertices = np.vstack([vertices, 0.5*(vertices[unique_edges[:, 0]] + vertices[unique_edges[:, 1]])])
        midpoint_of = np.full(len(edges), -1)
        midpoint_of[np.where(long)[0]] = midpoints[inverse.ravel()]
        midpoint_of = midpoint_of.reshape(-1, 3)  # per face, the midpoint on edges 01, 12, 20
        new_faces = [faces[(midpoint_of < 0).all(axis=1)]]
        for face, marks in zip(faces[(midpoint_of >= 0).any(axis=1)], midpoint_of[(midpoint_of >= 0).any(axis=1)]):
            new_faces.append(_split_triangle(face, marks))
        faces = np.vstack(new_faces)
    return vertices, faces


def _split_triangle(face, marks):
    """Sub triangles of one triangle with midpoints marks on edges 01, 12, 20 (-1 when unsplit), same winding."""
    count = int(np.sum(marks >= 0))
    # rotate so the split edges come first
    for turn in range(3):
        order = [(turn + i) % 3 for i in range(3)]
        rotated, midpoints = face[order], marks[order]
        if (count == 1 and midpoints[0] >= 0) or (count == 2 and midpoints[2] < 0) or count == 3:
            break
    v0, v1, v2 = rotated
    m01, m12, m20 = midpoints
    if count == 1:
        return np.array([[v0, m01, v2], [m01, v1, v2]])
    if count == 2:
        return np.array([[m01, v1, m12], [v0, m01, m12], [v0, m12, v2]])
    return np.array([[v0, m01, m20], [m01, v1, m12], [m20, m12, v2], [m01, m12, m20]])


def deform(vertices, faces, push):
    """Refined and pushed copy of a surface in mm, with the number of triangles whose facing flipped."""
    if np.linalg.norm(vertices - push.center, axis=1).min() < 1e-6:
        raise ValueError("the tumor center lies on the airway wall")
    vertices, faces = refine_for_push(vertices, faces, push)
    moved = push(vertices)
    before = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    after = np.cross(moved[faces[:, 1]] - moved[faces[:, 0]], moved[faces[:, 2]] - moved[faces[:, 0]])
    return moved, faces, int(np.sum(np.einsum("ij,ij->i", before, after) <= 0))


def grow_into_stl(stl_path, output_path, push):
    """Write the airway STL deformed by a push. STL files are in m."""
    vertices, faces = mesh.read_binary_stl(stl_path)
    moved, faces, flipped = deform(vertices*1e3, faces, push)
    mesh.write_binary_stl(output_path, moved*1e-3, faces)
    return moved, faces, flipped
