"""Surface extraction from a binary mask and binary STL output."""
import numpy as np
from scipy import ndimage
from skimage import measure


def mask_to_surface(mask, spacing, smoothing_sigma_voxels=1.0):
    """Marching cubes on a lightly smoothed mask, padded so the surface is closed. Returns vertices in mm."""
    padded = np.pad(mask.astype(np.float32), 1)
    if smoothing_sigma_voxels > 0:
        padded = ndimage.gaussian_filter(padded, smoothing_sigma_voxels)
    vertices, faces, _, _ = measure.marching_cubes(padded, level=0.5, spacing=tuple(spacing))
    vertices -= np.asarray(spacing)  # remove the padding offset
    return vertices, faces


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
