"""Slice by slice air segmentation of a CT volume, then 3D seeding and leak-limited growth of the airway."""
import numpy as np
from scipy import ndimage

AIR_THRESHOLD_HU = -950.0  # lowest threshold tried; lumen median is about -980 HU, lung parenchyma about -850 HU
MAX_THRESHOLD_HU = -800.0
THRESHOLD_STEP_HU = 10.0
LEAK_VOLUME_RATIO = 1.3  # airway volume growth between thresholds that signals leaking into the lungs
SEED_THRESHOLD_HU = -900.0
TRACHEA_AREA_MM2 = (60.0, 900.0)  # plausible axial cross section of an adult trachea
SEED_SEARCH_SLICES = 300  # slices searched downward from the head end
SEED_PERSIST_SLICES = 30  # a real trachea repeats at the same place in the slices below
SEED_PERSIST_MM = 2.0
CENTRAL_FRACTION = (0.25, 0.75)  # the trachea sits in the middle of the field of view


def body_interior_mask(volume):
    """True where air is not connected to the slice border, evaluated slice by slice at the loosest threshold."""
    interior = np.zeros(volume.shape, dtype=bool)
    for slice_index in range(volume.shape[2]):
        air = volume[:, :, slice_index] < MAX_THRESHOLD_HU
        labels, count = ndimage.label(air)
        if count == 0:
            continue
        border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        interior[:, :, slice_index] = ~np.isin(labels, border[border > 0])
    return interior


def tracheal_pockets(air_slice, spacing):
    """Round, central, tracheal-sized air pockets of one slice as (area in mm^2, center pixel, inside pixel)."""
    labels, count = ndimage.label(air_slice)
    pockets = []
    for label in range(1, count + 1):
        pixels = np.argwhere(labels == label)
        area = len(pixels)*spacing[0]*spacing[1]
        if not TRACHEA_AREA_MM2[0] <= area <= TRACHEA_AREA_MM2[1]:
            continue
        center = pixels.mean(axis=0)
        if not all(CENTRAL_FRACTION[0]*air_slice.shape[axis] < center[axis] < CENTRAL_FRACTION[1]*air_slice.shape[axis]
                   for axis in (0, 1)):
            continue
        extent = (pixels.max(axis=0) - pixels.min(axis=0) + 1)*spacing[:2]
        if extent.min()/extent.max() < 0.6:
            continue
        inside = pixels[np.argmin(np.linalg.norm(pixels - center, axis=1))]
        pockets.append((area, center, inside))
    return pockets


def find_trachea_seed(smoothed, interior, spacing):
    """Walk down from the head end to the first round air pocket that persists as a tube in the slices below."""
    top = smoothed.shape[2] - 1
    pocket_cache = {}

    def pockets_at(slice_index):
        if slice_index not in pocket_cache:
            air = (smoothed[:, :, slice_index] < SEED_THRESHOLD_HU) & interior[:, :, slice_index]
            pocket_cache[slice_index] = tracheal_pockets(air, spacing)
        return pocket_cache[slice_index]

    for slice_index in range(top, max(top - SEED_SEARCH_SLICES, SEED_PERSIST_SLICES) - 1, -1):
        for area, center, inside in pockets_at(slice_index):
            tube = True
            for below in range(1, SEED_PERSIST_SLICES + 1):
                tube = any(np.linalg.norm((other_center - center)*spacing[:2]) < SEED_PERSIST_MM
                           and 0.5 < other_area/area < 2.0 for other_area, other_center, _ in pockets_at(slice_index - below))
                if not tube:
                    break
            if tube:
                return int(inside[0]), int(inside[1]), slice_index - SEED_PERSIST_SLICES//2
    raise RuntimeError("no trachea-like air tube found near the head end")


def grow_airway(volume, spacing, start_threshold=AIR_THRESHOLD_HU):
    """Raise the air threshold step by step and stop just before the airway volume jumps (leak into the lungs)."""
    smoothed = ndimage.gaussian_filter(volume.astype(np.float32), 0.7)
    interior = body_interior_mask(smoothed)
    seed = find_trachea_seed(smoothed, interior, spacing)
    print(f"seed voxel {seed}")
    voxel_volume_ml = np.prod(spacing)*1e-3
    previous_mask, previous_count, previous_threshold = None, 0, start_threshold
    threshold = start_threshold
    while threshold <= MAX_THRESHOLD_HU:
        labels, _ = ndimage.label((smoothed < threshold) & interior)
        if labels[seed] > 0:
            mask = labels == labels[seed]
            count = int(mask.sum())
            print(f"threshold {threshold:.0f} HU: {count*voxel_volume_ml:.1f} mL")
            if previous_count and count > LEAK_VOLUME_RATIO*previous_count:
                break
            previous_mask, previous_count, previous_threshold = mask, count, threshold
        threshold += THRESHOLD_STEP_HU
    if previous_mask is None:
        raise RuntimeError("seed never fell inside the air mask")
    return previous_mask, previous_threshold
