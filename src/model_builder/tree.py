"""Cut an airway tree a set distance along the airway past the carina and extrude every cut end out of the domain.

Distances are geodesic, measured inside the airway from the top of the trachea, so each branch is cut at the same depth
whatever its direction. Each cut end is extruded along the grid axis closest to its own direction, which gives every outlet
a straight tube that leaves the domain through a face. Upward pointing ends go out sideways, so only the trachea crosses
the top face, which is the inlet.
"""
import numpy as np
from scipy import ndimage
from skimage.graph import MCP_Geometric

SHELL_MM = 2.0              # width of the distance bins used to find the carina and the cut ends
MIN_END_AREA_MM2 = 12.0     # cut ends narrower than about 4 mm stay closed instead of becoming needle thin outlets
MIN_CHILD_FRACTION = 0.15   # a branch has split when two pieces each carry at least this share of its section
MIN_PIECE_VOXELS = 30       # smaller pieces in a distance band are side twigs, not branches
DIRECTION_BACK_MM = (6.0, 12.0)  # the branch direction comes from the airway this far back from each cut
DIRECTION_RADIUS_MM = 15.0  # and within this distance of the cut end, so neighboring branches do not mix in
SLAB_MM = 3.0               # the airway this far behind a cut is swept outward, which carries the whole branch section
MIN_ALIGNMENT = 0.5         # an extrusion axis more than 60 degrees off the branch would sweep a thin flat sheet, so the branch stays closed

# extrusion axes: index, sign; +z is left out because the top face is the inlet
AXES = {"-x": (0, -1), "+x": (0, 1), "-y": (1, -1), "+y": (1, 1), "-z": (2, -1)}


def geodesic_distance(mask, spacing, start_slice):
    """Distance in mm inside the airway from the airway voxels of the start slice."""
    costs = np.where(mask, 1.0, np.inf)
    starts = [tuple(index) + (start_slice,) for index in np.argwhere(mask[:, :, start_slice])]
    distance, _ = MCP_Geometric(costs, sampling=tuple(spacing)).find_costs(starts)
    distance[~mask] = np.inf
    return distance


def carina_distance(mask, distance, search_mm=300.0):
    """The first distance band where the airway has split into two sizable pieces that stay split.

    Each band is labelled inside its own small bounding box, which keeps this fast on a full CT volume.
    """
    voxels = np.argwhere(mask)
    values = distance[tuple(voxels.T)]
    counts = []
    for start in np.arange(0.0, min(search_mm, values.max()), SHELL_MM):
        band = voxels[(values >= start) & (values < start + SHELL_MM)]
        if len(band) == 0:
            counts.append((start, 0)); continue
        low = band.min(axis=0)
        local = np.zeros(band.max(axis=0) - low + 1, dtype=bool)
        local[tuple((band - low).T)] = True
        labels, _ = ndimage.label(local)
        counts.append((start, int(np.sum(np.bincount(labels.ravel())[1:] >= 30))))
    for i, (start, count) in enumerate(counts[:-3]):
        if count >= 2 and all(c >= 2 for _, c in counts[i + 1:i + 4]):
            return start
    raise RuntimeError("no carina found along the airway")


def generation_cut(mask, distance, carina, length_past_split_mm):
    """Distance at which each airway voxel is cut: length_past_split_mm past the first split after the carina on its branch.

    The airway is followed outward in distance bands. Each piece of a band inherits from the piece it touches in the band
    before, and when a piece divides into two sizable children both children record that split distance.
    """
    voxels = np.argwhere(mask)
    values = distance[tuple(voxels.T)]
    cut_at = np.full(mask.shape, np.inf)
    previous = None  # (label volume of the previous band in its own box, box corner, piece split distances)
    for start in np.arange(carina, values.max() + SHELL_MM, SHELL_MM):
        band = voxels[(values >= start) & (values < start + SHELL_MM)]
        if len(band) == 0:
            break
        low = band.min(axis=0) - 1
        local = np.zeros(band.max(axis=0) - low + 2, dtype=bool)
        local[tuple((band - low).T)] = True
        labels, count = ndimage.label(local, structure=np.ones((3, 3, 3)))
        sizes = np.bincount(labels.ravel())
        split = {}
        parents = {}
        for piece in range(1, count + 1):
            parent = None
            if previous is not None:
                old_labels, old_low, _ = previous
                # parent: the previous band piece touching this one, found through the voxels of this piece shifted by one
                coords = np.argwhere(labels == piece) + low - old_low
                hits = []
                for shift in ((0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    c = coords + shift
                    ok = np.all((c >= 0) & (c < old_labels.shape), axis=1)
                    hits.extend(old_labels[tuple(c[ok].T)])
                hits = [h for h in hits if h > 0]
                parent = max(set(hits), key=hits.count) if hits else None
            parents[piece] = parent
        for piece in range(1, count + 1):
            parent = parents[piece]
            inherited = previous[2].get(parent, None) if (previous is not None and parent is not None) else None
            siblings = [q for q in range(1, count + 1) if parents[q] == parent and parent is not None and sizes[q] >= MIN_PIECE_VOXELS]
            family = sum(sizes[q] for q in siblings)
            divided = len(siblings) >= 2 and all(sizes[q] >= MIN_CHILD_FRACTION*family for q in siblings)
            if inherited is None and divided and start > carina + 2*SHELL_MM:
                inherited = start  # first split after the carina on this branch
            split[piece] = inherited
            if inherited is not None:
                coords = np.argwhere(labels == piece) + low
                cut_at[tuple(coords.T)] = inherited + length_past_split_mm
        previous = (labels, low, split)
    return cut_at


def cut_and_extrude(mask, spacing, top_slice, length_past_split_mm, extension_mm):
    """Keep the airway from top_slice down to length_past_split_mm past the first split after the carina on each branch,
    then extrude every cut end. Returns the new mask, the voxels padded on every side, the outlets, the carina distance, and the
    CT voxel index that the first unpadded voxel came from.
    """
    mask = mask.copy()
    mask[:, :, top_slice + 1:] = False
    labels, _ = ndimage.label(mask)
    mask = labels == labels[tuple(np.argwhere(mask[:, :, top_slice])[0]) + (top_slice,)]
    # work inside the airway bounding box, the rest of the CT volume is empty
    low, high = np.argwhere(mask).min(axis=0), np.argwhere(mask).max(axis=0) + 1
    mask = mask[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    top_slice -= low[2]
    distance = geodesic_distance(mask, spacing, top_slice)
    carina = carina_distance(mask, distance)
    cut_at = generation_cut(mask, distance, carina, length_past_split_mm)
    keep = mask & (distance <= cut_at)
    labels, _ = ndimage.label(keep)
    keep = labels == labels[tuple(np.argwhere(keep[:, :, top_slice])[0]) + (top_slice,)]

    # crop to the kept tree and pad every side so the extrusions have room
    lower = np.argwhere(keep).min(axis=0)
    upper = np.argwhere(keep).max(axis=0) + 1
    pad = [int(np.ceil(extension_mm/s)) for s in spacing]
    box = tuple(slice(lo, hi) for lo, hi in zip(lower, upper))
    keep, distance, removed = keep[box], distance[box], (mask & ~keep)[box]
    keep = np.pad(keep, [(p, p) for p in pad])
    distance = np.pad(distance, [(p, p) for p in pad], constant_values=np.inf)
    removed = np.pad(removed, [(p, p) for p in pad])

    # cut ends: kept voxels touching airway that was cut away
    # an angled cut leaves a thin, broken shell, so close it and connect through edges and corners to get one end per branch
    contact = keep & ndimage.binary_dilation(removed)
    ends, count = ndimage.label(ndimage.binary_closing(contact, structure=np.ones((3, 3, 3))) & keep, structure=np.ones((3, 3, 3)))
    extruded = keep.copy()
    outlets = []
    face_area = spacing[0]*spacing[1]
    for end in range(1, count + 1):
        voxels = np.argwhere(ends == end)
        if len(voxels)*face_area < MIN_END_AREA_MM2:
            continue
        center = voxels.mean(axis=0)
        end_distance = float(np.median(distance[tuple(voxels.T)]))
        band = keep & (distance > end_distance - DIRECTION_BACK_MM[1]) & (distance < end_distance - DIRECTION_BACK_MM[0])
        band_voxels = np.argwhere(band)
        near = band_voxels[np.linalg.norm((band_voxels - center)*spacing, axis=1) < DIRECTION_RADIUS_MM]
        if len(near) == 0:
            continue
        direction = (center - near.mean(axis=0))*spacing
        # the slab is only the piece of airway behind this cut that touches it, never a neighboring branch
        slab_labels, _ = ndimage.label(keep & (distance > end_distance - SLAB_MM) & (distance <= end_distance + SHELL_MM),
                                       structure=np.ones((3, 3, 3)))
        touching = np.unique(slab_labels[tuple(voxels.T)])
        slab = np.argwhere(np.isin(slab_labels, touching[touching > 0]))
        # best aligned axis whose straight tube does not run into another part of the airway
        own = np.zeros(keep.shape, dtype=bool)
        own[tuple(np.argwhere(keep & (np.linalg.norm(np.indices(keep.shape).reshape(3, -1).T.reshape(keep.shape + (3,))
                                                          - center, axis=-1) < DIRECTION_RADIUS_MM/min(spacing))).T)] = True
        for name in sorted(AXES, key=lambda key: -AXES[key][1]*direction[AXES[key][0]]):
            axis, sign = AXES[name]
            if AXES[name][1]*direction[AXES[name][0]] < MIN_ALIGNMENT*np.linalg.norm(direction):
                name = None
                break
            if not _sweep_hits(keep & ~own, slab, axis, sign):
                break
        else:
            name = None
        if name is None:
            continue
        _sweep(extruded, slab, axis, sign)
        outlets.append((name, center, len(voxels)*face_area, direction/np.linalg.norm(direction)))

    # the trachea leaves through the top face
    top = np.argwhere(keep[:, :, -pad[2] - 1])
    _sweep(extruded, np.column_stack([top, np.full(len(top), keep.shape[2] - pad[2] - 1)]), 2, 1)
    offset = low + lower  # CT voxel index of the first voxel before padding
    return extruded, pad, outlets, carina, offset


def _sweep(mask, voxels, axis, sign):
    """Translate a set of voxels step by step along one axis until it leaves the array, marking every position."""
    shifted = voxels.copy()
    while True:
        shifted[:, axis] += sign
        inside = (shifted[:, axis] >= 0) & (shifted[:, axis] < mask.shape[axis])
        if not inside.any():
            return
        mask[tuple(shifted[inside].T)] = True


def _sweep_hits(obstacles, voxels, axis, sign):
    """Whether translating the voxels along one axis to the edge of the array ever lands on an obstacle."""
    shifted = voxels.copy()
    while True:
        shifted[:, axis] += sign
        inside = (shifted[:, axis] >= 0) & (shifted[:, axis] < obstacles.shape[axis])
        if not inside.any():
            return False
        if obstacles[tuple(shifted[inside].T)].any():
            return True
