"""Extract an airway STL for every patient of a CT data set and tabulate measures that flag narrowing or irregular shape.

Run with: build/venv/bin/python -m model_builder.bank <data dir> <output dir> from src, where the data dir holds one
folder per patient with <id>_CT_HR.nii.gz and, when present, <id>_CT_HR_label_airways.nii.gz as in AeroPath.
"""
import argparse
import csv
import json
import os
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

from . import extract, mesh, segment

MIN_BRANCH_AREA_MM2 = 20.0      # an axial component this large counts as an airway branch when finding the carina
CARINA_PERSIST_MM = 5.0         # two branches must persist this far below a slice for it to be the carina
TRACHEA_END_MARGIN_MM = 10.0    # trimmed from both ends of the trachea so the larynx and carina do not skew the measures
LOCAL_WINDOW_MM = 25.0          # a narrowing is measured against the median area within this distance above and below
LOCAL_END_MM = 15.0             # and not within this distance of either end of the measured trachea


def trachea_profile(mask, spacing):
    """Axial area of the largest airway piece per slice from the head end down to the carina, in mm^2."""
    pixel_area = spacing[0]*spacing[1]
    slices = np.where(mask.any(axis=(0, 1)))[0]
    top = slices[-1]
    persist = int(round(CARINA_PERSIST_MM/spacing[2]))
    branch_counts, largest_areas = {}, {}
    for z in range(top, slices[0] - 1, -1):
        labels, count = ndimage.label(mask[:, :, z])
        areas = np.bincount(labels.ravel())[1:]*pixel_area if count else np.zeros(0)
        branch_counts[z] = int(np.sum(areas >= MIN_BRANCH_AREA_MM2))
        largest_areas[z] = float(areas.max()) if count else 0.0
    carina = None
    for z in range(top, slices[0] + persist - 1, -1):
        if all(branch_counts.get(z - below, 0) >= 2 for below in range(persist)):
            carina = z
            break
    if carina is None:
        carina = slices[0]
    margin = int(round(TRACHEA_END_MARGIN_MM/spacing[2]))
    heights = np.arange(carina + margin, top - margin + 1)
    return heights, np.array([largest_areas[z] for z in heights]), carina


def process(job):
    patient, data_dir, output_dir = job
    ct_path = os.path.join(data_dir, patient, f"{patient}_CT_HR.nii.gz")
    label_path = os.path.join(data_dir, patient, f"{patient}_CT_HR_label_airways.nii.gz")
    row = {"patient": int(patient), "status": "ok"}
    try:
        volume, spacing = extract.load_volume(ct_path)
        mask, threshold = segment.grow_airway(volume, spacing)
        row.update(threshold_hu=threshold, airway_volume_ml=mask.sum()*np.prod(spacing)*1e-3,
                   voxel_mm=" x ".join(f"{s:.2f}" for s in spacing))
        if os.path.exists(label_path):
            labels = extract.load_volume(label_path)[0] > 0
            row["dice_vs_labels"] = 2.0*np.logical_and(mask, labels).sum()/(mask.sum() + labels.sum())
            row["label_volume_ml"] = labels.sum()*np.prod(spacing)*1e-3

        heights, areas, carina = trachea_profile(mask, spacing)
        if len(areas) > 0:
            median = float(np.median(areas))
            narrowest = int(np.argmin(areas))
            row.update(trachea_length_mm=len(areas)*spacing[2] + 2*TRACHEA_END_MARGIN_MM, trachea_median_area_mm2=median,
                       trachea_min_area_mm2=float(areas[narrowest]), area_variation_percent=100.0*float(np.std(areas)/np.mean(areas)))
            # local narrowing: each height against the median of its neighbors, away from the trimmed ends where the airway tapers
            height_mm = (heights - carina)*spacing[2]
            inner = (height_mm > height_mm.min() + LOCAL_END_MM) & (height_mm < height_mm.max() - LOCAL_END_MM)
            ratios = np.full(len(areas), np.inf)
            for i in np.where(inner)[0]:
                ratios[i] = areas[i]/np.median(areas[np.abs(height_mm - height_mm[i]) <= LOCAL_WINDOW_MM])
            if np.isfinite(ratios).any():
                i = int(np.argmin(ratios))
                row.update(local_narrowing_percent=100.0*(1.0 - ratios[i]), local_narrowing_mm_above_carina=float(height_mm[i]))
        np.savez(os.path.join(output_dir, f"patient_{int(patient):02d}_profile.npz"), heights_mm=(heights - carina)*spacing[2], areas_mm2=areas)

        vertices, faces = mesh.mask_to_surface(mask, spacing, 1.0)
        vertices, faces = mesh.prune_small_components(vertices, faces, 0.01)
        faces = mesh.orient_outward(vertices, faces)
        vertices -= vertices.mean(axis=0)
        stl_path = os.path.join(output_dir, f"patient_{int(patient):02d}_airway.stl")
        mesh.write_binary_stl(stl_path, vertices*1e-3, faces)
        extent = vertices.max(axis=0) - vertices.min(axis=0)
        row.update(stl=os.path.basename(stl_path), triangles=len(faces), extent_mm=" x ".join(f"{e:.0f}" for e in extent))
    except Exception as error:  # a patient that fails is reported in the table, the rest still run
        row.update(status=f"failed: {error}")
    print(f"patient {patient}: {row['status']}", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    patients = sorted((p for p in os.listdir(args.data_dir) if p.isdigit()), key=int)
    with Pool(args.workers) as pool:
        rows = pool.map(process, [(p, args.data_dir, args.output_dir) for p in patients])
    columns = ["patient", "status", "stl", "triangles", "extent_mm", "voxel_mm", "threshold_hu", "airway_volume_ml", "label_volume_ml",
               "dice_vs_labels", "trachea_length_mm", "trachea_median_area_mm2", "trachea_min_area_mm2", "local_narrowing_percent",
               "local_narrowing_mm_above_carina", "area_variation_percent"]
    with open(os.path.join(args.output_dir, "airway_bank.csv"), "w", newline="") as table:
        writer = csv.DictWriter(table, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in row.items() if k in columns})
    with open(os.path.join(args.output_dir, "airway_bank.json"), "w") as raw:
        json.dump(rows, raw, indent=1, default=float)


if __name__ == "__main__":
    main()
