"""Command line tool: CT NIfTI volume to airway STL."""
import argparse
import nibabel as nib
import numpy as np

from . import mesh, segment


def load_volume(path):
    """Load a NIfTI file in RAS+ order so +z is toward the head. Returns data and voxel spacing in mm."""
    image = nib.as_closest_canonical(nib.load(path))
    return np.asanyarray(image.dataobj), np.array(image.header.get_zooms()[:3])


def extract_airway(ct_path, output_path, label_path=None, threshold=None,
                   smoothing=1.0, units_per_mm=1.0e-3, center=True, prune_fraction=0.01,
                   crop_mm=None, extension_mm=20.0, thin_branch_mm=0.0):
    volume, spacing = load_volume(ct_path)
    threshold = segment.AIR_THRESHOLD_HU if threshold is None else threshold
    if label_path is None:
        mask, final_threshold = segment.grow_airway(volume, spacing, threshold)
        print(f"final threshold {final_threshold:.0f} HU")
    else:
        mask = load_volume(label_path)[0] > 0
    voxel_volume_ml = np.prod(spacing)*1e-3
    print(f"airway volume {mask.sum()*voxel_volume_ml:.1f} mL, {mask.sum()} voxels")

    mask = mesh.remove_thin_branches(mask, spacing, thin_branch_mm)
    if crop_mm is not None:
        mask = mesh.crop_and_extrude(mask, spacing, crop_mm, extension_mm)
    vertices, faces = mesh.mask_to_surface(mask, spacing, smoothing)
    if prune_fraction > 0:
        vertices, faces = mesh.prune_small_components(vertices, faces, prune_fraction)
    faces = mesh.orient_outward(vertices, faces)
    if center:
        vertices -= vertices.mean(axis=0)
    vertices *= units_per_mm
    mesh.write_binary_stl(output_path, vertices, faces)
    print(f"wrote {output_path}: {len(faces)} triangles, volume {mesh.signed_volume(vertices, faces):.3e} (units^3)")
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ct", help="CT volume, .nii or .nii.gz")
    parser.add_argument("output", help="output binary STL")
    parser.add_argument("--labels", help="airway label volume; skips thresholding and seeding")
    parser.add_argument("--threshold", type=float, default=segment.AIR_THRESHOLD_HU, help="air threshold in HU")
    parser.add_argument("--smoothing", type=float, default=1.0, help="gaussian sigma in voxels before surfacing")
    parser.add_argument("--units-per-mm", type=float, default=1.0e-3, help="STL length unit per mm, default meters")
    parser.add_argument("--prune", type=float, default=0.01, help="drop pieces under this fraction of the largest, 0 keeps all")
    parser.add_argument("--crop-mm", type=float, nargs=6, metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                        help="keep this box of the volume and extrude its z ends as open tubes for inflow and outflow")
    parser.add_argument("--thin-branch-mm", type=float, default=0.0, help="delete branches thinner than this radius")
    parser.add_argument("--extension-mm", type=float, default=20.0, help="length of each extruded end")
    parser.add_argument("--no-center", action="store_true", help="keep the scanner position instead of centering")
    args = parser.parse_args()
    extract_airway(args.ct, args.output, args.labels, args.threshold, args.smoothing, args.units_per_mm, not args.no_center, args.prune,
                   args.crop_mm, args.extension_mm, args.thin_branch_mm)


if __name__ == "__main__":
    main()
