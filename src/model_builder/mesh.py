"""Surface extraction from a binary mask and binary STL output."""
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
from skimage import measure


def mask_to_surface(mask, spacing, smoothing_sigma_voxels=1.0):
    """Marching cubes on a lightly smoothed mask, padded so the surface is closed. Returns vertices in mm."""
    padded = np.pad(mask.astype(np.float32), 1)
    if smoothing_sigma_voxels > 0:
        padded = ndimage.gaussian_filter(padded, smoothing_sigma_voxels)
    vertices, faces, _, _ = measure.marching_cubes(padded, level=0.5, spacing=tuple(spacing))
    vertices -= np.asarray(spacing)  # remove the padding offset
    return vertices, faces


def remove_thin_branches(mask, spacing, radius_mm):
    """Morphological opening that deletes airway pieces thinner than the radius, then keeps the largest connected piece."""
    iterations = int(round(radius_mm/np.mean(spacing[:2])))
    if iterations < 1:
        return mask
    opened = ndimage.binary_opening(mask, structure=ndimage.generate_binary_structure(3, 1), iterations=iterations)
    labels, count = ndimage.label(opened)
    return labels == (np.argmax(np.bincount(labels.ravel())[1:]) + 1)


def crop_and_extrude(mask, spacing, crop_mm, extension_mm):
    """Keep the box crop_mm = (x0, x1, y0, y1, z0, z1) in volume coordinates and extrude its z ends straight into open tubes."""
    lower = [int(round(crop_mm[2*axis]/spacing[axis])) for axis in range(3)]
    upper = [int(round(crop_mm[2*axis + 1]/spacing[axis])) for axis in range(3)]
    mask = mask[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
    pad = int(round(extension_mm/spacing[2]))
    return np.pad(mask, ((0, 0), (0, 0), (pad, pad)), mode="edge")


def prune_small_components(vertices, faces, min_fraction):
    """Drop vertex-connected pieces with fewer triangles than min_fraction of the largest piece."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]]])
    graph = sparse.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices),)*2)
    count, vertex_labels = connected_components(graph, directed=False)
    triangles_per_piece = np.bincount(vertex_labels[faces[:, 0]], minlength=count)
    keep_piece = triangles_per_piece >= min_fraction*triangles_per_piece.max()
    keep_face = keep_piece[vertex_labels[faces[:, 0]]]
    used_vertices, new_indices = np.unique(faces[keep_face], return_inverse=True)
    print(f"pruned {count - keep_piece.sum()} of {count} pieces, {(~keep_face).sum()} triangles")
    return vertices[used_vertices], new_indices.reshape(-1, 3)


def signed_volume(vertices, faces):
    corners = vertices[faces]
    return np.einsum("ij,ij->", corners[:, 0], np.cross(corners[:, 1], corners[:, 2]))/6.0


def orient_outward(vertices, faces):
    """Make triangle winding give positive volume, so normals point out of the solid."""
    if signed_volume(vertices, faces) < 0:
        faces = faces[:, ::-1]
    return faces


def write_binary_stl(path, vertices, faces):
    corners = vertices[faces].astype(np.float32)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0).astype(np.float32)
    record = np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attribute", "<u2")])
    triangles = np.zeros(len(faces), dtype=record)
    triangles["normal"], triangles["corners"] = normals, corners
    with open(path, "wb") as stl_file:
        stl_file.write(b"model_builder".ljust(80, b"\0"))
        stl_file.write(np.uint32(len(faces)).tobytes())
        stl_file.write(triangles.tobytes())


def read_binary_stl(path):
    """Vertices and faces of a binary STL, with corners shared between triangles merged into one vertex."""
    record = np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attribute", "<u2")])
    with open(path, "rb") as stl_file:
        stl_file.seek(80)
        count = int(np.frombuffer(stl_file.read(4), dtype=np.uint32)[0])
        triangles = np.frombuffer(stl_file.read(count*record.itemsize), dtype=record)
    vertices, faces = np.unique(triangles["corners"].reshape(-1, 3), axis=0, return_inverse=True)
    return vertices.astype(np.float64), faces.reshape(-1, 3)
