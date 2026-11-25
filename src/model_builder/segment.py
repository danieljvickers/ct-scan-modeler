"""Slice by slice air segmentation of a CT volume, then 3D seeding and leak-limited growth of the airway."""
import numpy as np
from scipy import ndimage

AIR_THRESHOLD_HU = -950.0  # lowest threshold tried; lumen median is about -980 HU, lung parenchyma about -850 HU
MAX_THRESHOLD_HU = -800.0
THRESHOLD_STEP_HU = 10.0
LEAK_VOLUME_RATIO = 1.3  # airway volume growth between thresholds that signals leaking into the lungs
SEED_THRESHOLD_HU = -900.0
TRACHEA_AREA_MM2 = (60.0, 900.0)  # plausible axial cross section of an adult trachea
SEED_SEARCH_MM = 150.0  # searched downward from the head end
SEED_PERSIST_LENGTH_MM = 15.0  # a real trachea continues as a tube for at least this length below the seed
SEED_DRIFT_MM = 2.0  # allowed shift of the tube center between neighboring slices, plus half a slice for the tilt
CENTRAL_FRACTION = (0.25, 0.75)  # the trachea sits in the middle of the field of view
RECOVER_THRESHOLD_HU = -500.0  # streak artifacts can lift air inside the lumen well above the growth threshold
RECOVER_MIN_GROWTH = 1.5  # below this the loose region is only the partial volume ring of a clean slice, nothing is missing
RECOVER_MAX_GROWTH = 3.0  # a recovered region may be at most this many times the airway it grew from
RECOVER_MAX_AREA_MM2 = 1500.0  # and never larger than a wide trachea, which keeps it out of the lungs
RECOVER_MIN_AIRWAY_MM2 = 80.0  # only trachea and main bronchus sections, small bronchi sit in lung that is also below the loose threshold


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
    """Walk down from the head end to the first round air pocket that continues as a tube, followed slice by slice."""
    top = smoothed.shape[2] - 1
    persist_slices = max(1, int(round(SEED_PERSIST_LENGTH_MM/spacing[2])))
    search_slices = int(round(SEED_SEARCH_MM/spacing[2]))
    drift = SEED_DRIFT_MM + 0.5*spacing[2]
    pocket_cache = {}

    def pockets_at(slice_index):
        if slice_index not in pocket_cache:
            air = (smoothed[:, :, slice_index] < SEED_THRESHOLD_HU) & interior[:, :, slice_index]
            pocket_cache[slice_index] = tracheal_pockets(air, spacing)
        return pocket_cache[slice_index]

    for slice_index in range(top, max(top - search_slices, persist_slices) - 1, -1):
        for area, center, inside in pockets_at(slice_index):
            current_area, current_center, tube = area, center, True
            for below in range(1, persist_slices + 1):
                matches = [(np.linalg.norm((other_center - current_center)*spacing[:2]), other_area, other_center)
                           for other_area, other_center, _ in pockets_at(slice_index - below)
                           if 0.5 < other_area/current_area < 2.0]
                matches = [m for m in matches if m[0] < drift]
                if not matches:
                    tube = False
                    break
                _, current_area, current_center = min(matches, key=lambda m: m[0])
            if tube:
                return int(inside[0]), int(inside[1]), slice_index - persist_slices//2
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
    return recover_streaked_air(smoothed, previous_mask, spacing), previous_threshold


def recover_streaked_air(smoothed, mask, spacing):
    """Per axial slice, widen the airway to each loose-threshold air region it touches, when that region stays airway sized."""
    pixel_area = spacing[0]*spacing[1]
    recovered = mask.copy()
    for z in np.where(mask.any(axis=(0, 1)))[0]:
        loose_labels, count = ndimage.label(smoothed[:, :, z] < RECOVER_THRESHOLD_HU)
        if count == 0:
            continue
        airway = mask[:, :, z]
        for region_label in np.unique(loose_labels[airway]):
            if region_label == 0:
                continue
            region = loose_labels == region_label
            region_area = region.sum()*pixel_area
            airway_area = (region & airway).sum()*pixel_area  # every airway piece inside it, a streak can split the lumen
            growth = region_area/max(airway_area, 1e-12)
            if airway_area >= RECOVER_MIN_AIRWAY_MM2 and RECOVER_MIN_GROWTH <= growth <= RECOVER_MAX_GROWTH and region_area <= RECOVER_MAX_AREA_MM2:
                recovered[:, :, z] |= region
    return recovered
