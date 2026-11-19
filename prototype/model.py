"""STL reading, transforms, and 2D geometry queries, following MFC's m_model."""

import os

import numpy as np

POINT_CHUNK_SIZE = 4096  # query points per vectorized block, bounds memory at chunk*edges
SURFACE_EPS = 1.0e-10    # relative to the model size, distance ties and on-surface points


def _read_stl_binary(filepath):
    with open(filepath, "rb") as stl_file:
        stl_file.read(80)
        num_triangles = int(np.frombuffer(stl_file.read(4), dtype="<u4")[0])
        expected_size = 84 + 50*num_triangles
        assert os.path.getsize(filepath) == expected_size, f"size {os.path.getsize(filepath)} is not {expected_size}"
        record = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
        return np.fromfile(stl_file, dtype=record, count=num_triangles)["vertices"].astype(np.float64)


def _read_stl_ascii(filepath):
    with open(filepath, "r") as stl_file:
        vertices = [line.split()[1:4] for line in stl_file if line.strip().startswith("vertex")]
    assert vertices and len(vertices) % 3 == 0, f"found {len(vertices)} vertices, not a positive multiple of 3"
    return np.array(vertices, dtype=np.float64).reshape(-1, 3, 3)


def read_stl(filepath):
    """Return triangle vertices (num_triangles, 3, 3), trying binary then ASCII."""
    failures = []
    for reader in (_read_stl_binary, _read_stl_ascii):
        try:
            return reader(filepath)
        except Exception as error:
            failures.append(f"{reader.__name__}: {error}")
    raise RuntimeError(f"ERROR :: could not read {filepath}\n  " + "\n  ".join(failures))


def transform_triangles(triangles, scale, rotate, translate):
    """Scale and rotate about the bounding box center, then translate, as MFC does."""
    center = 0.5*(triangles.reshape(-1, 3).min(axis=0) + triangles.reshape(-1, 3).max(axis=0))
    cos_x, sin_x = np.cos(rotate[0]), np.sin(rotate[0])
    cos_y, sin_y = np.cos(rotate[1]), np.sin(rotate[1])
    cos_z, sin_z = np.cos(rotate[2]), np.sin(rotate[2])
    rotation_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
    rotation_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    rotation_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
    rotation = rotation_x @ rotation_y @ rotation_z
    return ((triangles - center)*np.asarray(scale)) @ rotation.T + center + np.asarray(translate)


def boundary_edges_2d(triangles):
    """Return the outline edges (num_edges, 2, 2) of a flat triangulation, oriented counterclockwise."""
    vertices = triangles[:, :, :2].copy()

    # orient every triangle counterclockwise so the outline winds consistently
    edge_1, edge_2 = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
    clockwise = edge_1[:, 0]*edge_2[:, 1] - edge_1[:, 1]*edge_2[:, 0] < 0
    vertices[clockwise] = vertices[clockwise][:, ::-1]

    # an edge used by exactly one triangle lies on the outline
    edges = np.concatenate([vertices[:, [0, 1]], vertices[:, [1, 2]], vertices[:, [2, 0]]])
    tolerance = 1.0e-9*np.ptp(vertices.reshape(-1, 2), axis=0).max()
    quantized = np.round(edges/tolerance).astype(np.int64).reshape(-1, 4)
    start_after_end = (quantized[:, 0] > quantized[:, 2]) | ((quantized[:, 0] == quantized[:, 2]) & (quantized[:, 1] > quantized[:, 3]))
    undirected = np.where(start_after_end[:, None], quantized[:, [2, 3, 0, 1]], quantized)
    _, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
    return edges[counts[inverse.ravel()] == 1]


def winding_number_2d(edges, points):
    """Generalized winding number, Jacobson et al. (2013): ~1 inside and ~0 outside."""
    winding = np.empty(len(points))
    for start in range(0, len(points), POINT_CHUNK_SIZE):
        chunk = points[start:start + POINT_CHUNK_SIZE, None, :]
        to_start, to_end = edges[None, :, 0] - chunk, edges[None, :, 1] - chunk
        cross = to_start[..., 0]*to_end[..., 1] - to_start[..., 1]*to_end[..., 0]
        dot = (to_start*to_end).sum(axis=-1)
        winding[start:start + POINT_CHUNK_SIZE] = np.arctan2(cross, dot).sum(axis=1)/(2.0*np.pi)
    return winding


def boundary_distances_and_normals_2d(edges, points, tie_tolerance):
    """Distance to the outline and the unit direction toward it, averaged over edges tied for closest.

    Averaging keeps the normal mirror symmetric when a point is equidistant from two edges, e.g. on a diagonal.
    """
    distances = np.empty(len(points))
    normals = np.empty_like(points)
    edge_vectors = edges[:, 1] - edges[:, 0]
    edge_lengths_squared = np.maximum((edge_vectors**2).sum(axis=1), np.finfo(float).tiny)
    outward_normals = outward_edge_normals_2d(edges)
    for start in range(0, len(points), POINT_CHUNK_SIZE):
        chunk = points[start:start + POINT_CHUNK_SIZE, None, :]
        fraction = np.clip(((chunk - edges[None, :, 0])*edge_vectors).sum(axis=-1)/edge_lengths_squared, 0.0, 1.0)
        offsets = edges[None, :, 0] + fraction[..., None]*edge_vectors - chunk
        candidate_distances = np.linalg.norm(offsets, axis=-1)
        nearest_distances = candidate_distances.min(axis=1)

        # direction to each candidate, or the edge normal for a point on the surface
        on_surface = candidate_distances[..., None] <= tie_tolerance
        directions = np.where(on_surface, outward_normals[None], offsets/np.maximum(candidate_distances, tie_tolerance)[..., None])
        is_tied = candidate_distances <= nearest_distances[:, None] + tie_tolerance
        summed = (directions*is_tied[..., None]).sum(axis=1)
        distances[start:start + POINT_CHUNK_SIZE] = nearest_distances
        normals[start:start + POINT_CHUNK_SIZE] = summed/np.linalg.norm(summed, axis=1, keepdims=True)
    return distances, normals


def outward_edge_normals_2d(edges):
    """Unit normals pointing out of a counterclockwise outline."""
    edge_vectors = edges[:, 1] - edges[:, 0]
    normals = np.column_stack([edge_vectors[:, 1], -edge_vectors[:, 0]])
    return normals/np.linalg.norm(normals, axis=1, keepdims=True)


class Model2D:
    """A 2D STL outline placed in the domain."""

    def __init__(self, stl_parameters, case_dir):
        filepath = os.path.join(case_dir, stl_parameters["model_filepath"])
        triangles = transform_triangles(read_stl(filepath),
                                        [stl_parameters[f"model_scale({k})"] for k in range(1, 4)],
                                        [stl_parameters[f"model_rotate({k})"] for k in range(1, 4)],
                                        [stl_parameters[f"model_translate({k})"] for k in range(1, 4)])
        self.filepath = filepath
        self.threshold = stl_parameters["model_threshold"]
        self.edges = boundary_edges_2d(triangles)
        self.bounding_box = np.array([self.edges.reshape(-1, 2).min(axis=0), self.edges.reshape(-1, 2).max(axis=0)])

    def is_inside(self, points):
        return winding_number_2d(self.edges, points) > self.threshold

    def levelset(self, points):
        """Distance to the outline and the outward unit normal for points inside the body."""
        return boundary_distances_and_normals_2d(self.edges, points, SURFACE_EPS*np.ptp(self.bounding_box, axis=0).max())
