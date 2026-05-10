"""
detect.py — Coin Detection Pipeline

Hough circle detection with adaptive parameter selection, a four-stage
false-positive filter, and multi-step deduplication via NMS, arc-coverage
merge, cluster-wrapper removal, and nested-circle removal.
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# NMS and geometric post-processing
# ──────────────────────────────────────────────────────────────────────────────

def non_maximum_suppression(circles: list, overlap_thresh: float = 0.4) -> list:
    """Sort ascending by radius; suppress a candidate if it overlaps a kept
    circle by more than overlap_thresh of the summed radii.  Size-ratio guard
    0.75 prevents a small bimetallic inner ring from suppressing its outer coin."""
    if not circles:
        return []
    circles = sorted(circles, key=lambda c: c[2])
    keep = []
    for (x1, y1, r1) in circles:
        suppressed = any(
            np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < (r1 + r2) * (1.0 - overlap_thresh)
            and r2 >= r1 * 0.75
            for (x2, y2, r2) in keep
        )
        if not suppressed:
            keep.append((x1, y1, r1))
    return keep


def _circle_overlap_frac(x1, y1, r1, x2, y2, r2) -> float:
    """Fraction of the smaller circle's area covered by the intersection."""
    d = float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
    r_s, r_l = min(r1, r2), max(r1, r2)
    if d >= r_s + r_l:
        return 0.0
    if d <= r_l - r_s:
        return 1.0
    alpha = 2.0 * np.arccos(np.clip((d**2 + r_l**2 - r_s**2) / (2.0 * d * r_l), -1, 1))
    beta  = 2.0 * np.arccos(np.clip((d**2 + r_s**2 - r_l**2) / (2.0 * d * r_s), -1, 1))
    area  = 0.5 * r_l**2 * (alpha - np.sin(alpha)) + 0.5 * r_s**2 * (beta - np.sin(beta))
    return float(area / (np.pi * r_s**2))


def _merge_by_arc_coverage(circles: list, edges: np.ndarray,
                            overlap_thresh: float = 0.40,
                            ratio_range: tuple = (0.70, 1.30)) -> list:
    """Post-NMS: when two similar-size circles overlap >overlap_thresh of the
    smaller area, keep whichever has higher arc-coverage and discard the other."""
    if len(circles) <= 1:
        return circles
    result = list(circles)
    changed = True
    while changed:
        changed = False
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                x1, y1, r1 = result[i]
                x2, y2, r2 = result[j]
                r_ratio = min(r1, r2) / max(r1, r2) if max(r1, r2) > 0 else 0
                if not (ratio_range[0] <= r_ratio <= ratio_range[1]):
                    continue
                if _circle_overlap_frac(x1, y1, r1, x2, y2, r2) < overlap_thresh:
                    continue
                cov1 = _arc_coverage(edges, x1, y1, r1)
                cov2 = _arc_coverage(edges, x2, y2, r2)
                keep_idx = i if cov1 >= cov2 else j
                drop_idx = j if keep_idx == i else i
                result[i] = result[keep_idx]
                result.pop(drop_idx)
                changed = True
                break
            if changed:
                break
    return result


def _remove_nested_circles(circles: list, proximity_ratio: float = 0.60) -> list:
    """Remove a circle that is nested inside a larger one.

    Three conditions (any one triggers removal):
      1. Centre proximity  — small circle's centre is within proximity_ratio × r_large.
      2. Geometric containment — dist + r_small < r_large × 1.05 (fully inside).
      3. High overlap — the small circle's area is > 70 % covered by the large circle.
         Catches edge-sitting FPs that are mostly inside a real coin but not fully so.
    """
    if len(circles) <= 1:
        return circles
    result = []
    for (x1, y1, r1) in circles:
        nested = False
        for (x2, y2, r2) in circles:
            if r2 <= r1:
                continue
            d = float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
            if (d < r2 * proximity_ratio                  # centre proximity
                    or d + r1 < r2 * 1.05                 # geometric containment
                    or _circle_overlap_frac(x1, y1, r1, x2, y2, r2) > 0.70):  # mostly inside
                nested = True
                break
        if not nested:
            result.append((x1, y1, r1))
    return result


def _remove_cluster_wrappers(circles: list, edges: np.ndarray = None) -> list:
    """Remove circles that contain 2+ clearly smaller circles (r_enc < 0.75×r)
    within 0.85×r of their centre — these are over-sized Hough artefacts.

    Two guards prevent removing a real (partially-occluded) coin:
      1. Arc-coverage guard: if the candidate's own arc coverage >= 0.38 it has
         a real circular edge and should not be discarded as a wrapper.
      2. Enclosed-overlap guard: if the 2 enclosed circles overlap each other by
         >30 % of the smaller area, they are both FPs from the same coin region —
         the enclosing circle is the real detection, so keep it.
    """
    if len(circles) <= 2:
        return circles
    result = []
    for (x1, y1, r1) in circles:
        inside = [
            (x2, y2, r2) for (x2, y2, r2) in circles
            if (x2, y2, r2) != (x1, y1, r1)
            and np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < r1 * 0.85
            and r2 < r1 * 0.75
        ]
        if len(inside) < 2:
            result.append((x1, y1, r1))
            continue

        # Guard 1 — real coin edge: arc coverage of the candidate itself
        if edges is not None and _arc_coverage(edges, x1, y1, r1) >= 0.38:
            result.append((x1, y1, r1))
            continue

        # Guard 2 — enclosed FPs: the two enclosed circles overlap each other
        if len(inside) >= 2:
            xa, ya, ra = inside[0]
            xb, yb, rb = inside[1]
            if _circle_overlap_frac(xa, ya, ra, xb, yb, rb) > 0.30:
                result.append((x1, y1, r1))
                continue

        # Confirmed wrapper artefact — drop it
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Hough circle detection
# ──────────────────────────────────────────────────────────────────────────────

def _hough_candidates(
    image: np.ndarray,
    min_radius: int = 30,
    max_radius: int = 250,
    bg_edge_density: float = 0.0,
) -> list:
    """Ensemble of HOUGH_GRADIENT and HOUGH_GRADIENT_ALT.

    GRADIENT_ALT uses a quality threshold in [0,1] and produces fewer, more
    precise circles — primary source.  HOUGH_GRADIENT with adaptive param2
    is kept as secondary to catch coins missed by ALT.
    """
    from src.preprocess import preprocess, _scale_for_processing

    small, scale = _scale_for_processing(image)
    smoothed = preprocess(small)
    min_r    = max(1, int(min_radius * scale))
    max_r    = max(min_r + 1, int(max_radius * scale))
    min_dist = max(1, int(60 * scale))

    all_raw = []

    # Primary: HOUGH_GRADIENT_ALT
    # param2 quality: 0.90 (plain) → 0.75 (textured)
    param2_alt = float(np.clip(0.90 - 0.15 * bg_edge_density / 0.20, 0.75, 0.90))
    circles_alt = cv2.HoughCircles(
        smoothed, cv2.HOUGH_GRADIENT_ALT, dp=1.5,
        minDist=min_dist, param1=300, param2=param2_alt,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles_alt is not None:
        all_raw += [(int(round(x / scale)), int(round(y / scale)), int(round(r / scale)))
                    for (x, y, r) in np.around(circles_alt[0]).astype(np.int32).tolist()]

    # Secondary: HOUGH_GRADIENT (only when ALT returns very few candidates)
    if len(all_raw) < 3:
        param2_grad = int(np.clip(45 + 20 * bg_edge_density / 0.20, 45, 65))
        circles_grad = cv2.HoughCircles(
            smoothed, cv2.HOUGH_GRADIENT, dp=1,
            minDist=min_dist, param1=40, param2=param2_grad,
            minRadius=min_r, maxRadius=max_r,
        )
        if circles_grad is not None:
            all_raw += [(int(round(x / scale)), int(round(y / scale)), int(round(r / scale)))
                        for (x, y, r) in np.around(circles_grad[0]).astype(np.int32).tolist()]

    return non_maximum_suppression(all_raw, overlap_thresh=0.5)


# ──────────────────────────────────────────────────────────────────────────────
# False-positive filter
# ──────────────────────────────────────────────────────────────────────────────

_N72 = 72
_A72 = np.linspace(0, 2 * np.pi, _N72, endpoint=False)
_C72 = np.cos(_A72)
_S72 = np.sin(_A72)


def _arc_coverage(edges, x, y, r, n_sectors=36, ring_width=8):
    h, w = edges.shape[:2]
    sectors_hit = np.zeros(n_sectors, dtype=bool)
    r_inner = max(r - ring_width, 1);  r_outer = r + ring_width
    y0 = max(0, y - r_outer);  y1 = min(h, y + r_outer + 1)
    x0 = max(0, x - r_outer);  x1 = min(w, x + r_outer + 1)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dist_sq = (ys - y) ** 2 + (xs - x) ** 2
    in_ring = (dist_sq >= r_inner ** 2) & (dist_sq <= r_outer ** 2)
    eyx = np.argwhere(in_ring & (edges[y0:y1, x0:x1] > 0))
    for ey, ex in eyx:
        angle = np.arctan2(ey - (y - y0), ex - (x - x0))
        sectors_hit[int((angle + np.pi) / (2 * np.pi) * n_sectors) % n_sectors] = True
    return float(sectors_hit.sum()) / n_sectors


def _edge_density_on_ring(edges, x, y, r, ring_width=8):
    h, w = edges.shape[:2]
    mo = np.zeros((h, w), np.uint8);  mi = np.zeros((h, w), np.uint8)
    cv2.circle(mo, (x, y), r + ring_width, 255, -1)
    cv2.circle(mi, (x, y), max(r - ring_width, 0), 255, -1)
    ring = cv2.subtract(mo, mi)
    ring_px = np.count_nonzero(ring)
    return np.count_nonzero(cv2.bitwise_and(edges, edges, mask=ring)) / ring_px if ring_px else 0.0


def _color_uniformity_inside(image, x, y, r):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (x, y), max(int(r * 0.80), 1), 255, -1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat_mask = (hsv[:, :, 1] > 12).astype(np.uint8) * 255
    hue_vals = hsv[:, :, 0][cv2.bitwise_and(mask, sat_mask) > 0]
    if len(hue_vals) >= 10:
        return 1.0 / (1.0 + float(np.std(hue_vals.astype(float))) / 30.0)
    v_vals = hsv[:, :, 2][mask > 0].astype(float)
    return 1.0 / (1.0 + float(np.std(v_vals)) / 40.0) if len(v_vals) > 0 else 0.0


def _edge_density_at_radius(edges, cx, cy, sample_r):
    H, W = edges.shape
    xs = np.clip(np.round(cx + sample_r * _C72).astype(np.int32), 0, W - 1)
    ys = np.clip(np.round(cy + sample_r * _S72).astype(np.int32), 0, H - 1)
    return float((edges[ys, xs] > 0).mean())


def _ring_edge_sharpness(edges, cx, cy, r):
    ring  = _edge_density_at_radius(edges, cx, cy, r)
    inner = _edge_density_at_radius(edges, cx, cy, max(r * 0.55, 1.0))
    outer = _edge_density_at_radius(edges, cx, cy, r * 1.55)
    bg = (inner + outer) * 0.5
    return ring / bg if bg > 1e-4 else ring * 100.0


def _adaptive_sharpness_threshold(bg_edge_density: float,
                                   base: float = 1.2) -> float:
    """Smooth: 1.0 + (base-1.0)×exp(−10×bg_tex). Plain=base, textured→1.0."""
    return 1.0 + (base - 1.0) * float(np.exp(-10.0 * bg_edge_density))


def filter_false_positives(
    image: np.ndarray,
    circles: list,
    edge_density_min: float     = 0.08,
    color_uniformity_min: float = 0.28,
    arc_coverage_min: float     = 0.35,
    ring_sharpness_min: float   = 1.5,
    bg_edge_density: float      = None,
) -> tuple:
    """Four-stage FP filter. Returns (valid_circles, edges)."""
    from src.preprocess import build_enhanced_edges

    if image.ndim < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    edges = build_enhanced_edges(image)

    if bg_edge_density is not None:
        ring_sharpness_min = _adaptive_sharpness_threshold(bg_edge_density,
                                                            ring_sharpness_min)

    valid = []
    for (x, y, r) in circles:
        x, y, r = int(x), int(y), int(r)
        arc_cov = _arc_coverage(edges, x, y, r)
        if arc_cov                                    < arc_coverage_min:      continue
        if _edge_density_on_ring(edges, x, y, r)     < edge_density_min:      continue
        if _color_uniformity_inside(image, x, y, r)  < color_uniformity_min:  continue
        if _ring_edge_sharpness(edges, x, y, r)      < ring_sharpness_min:    continue
        valid.append((x, y, r))

    return valid, edges


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def detect_coins(
    image: np.ndarray,
    border_px: int  = 60,
    min_radius: int = 25,
    max_radius: int = 250,
) -> list:
    """Detect coins. Returns list of (x, y, radius) tuples."""
    from src.preprocess import compute_background_edge_density

    bg_edge_density = compute_background_edge_density(image, border_px=border_px)
    candidates = _hough_candidates(
        image, min_radius=min_radius, max_radius=max_radius,
        bg_edge_density=bg_edge_density,
    )
    if not candidates:
        return []

    filtered, edges = filter_false_positives(
        image, candidates, bg_edge_density=bg_edge_density,
    )
    if not filtered:
        return []

    deduped = non_maximum_suppression(filtered, overlap_thresh=0.4)
    merged  = _merge_by_arc_coverage(deduped, edges)
    merged  = _remove_cluster_wrappers(merged, edges)
    return _remove_nested_circles(merged, proximity_ratio=0.60)


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_debug_candidates(image: np.ndarray, **kwargs) -> dict:
    from src.preprocess import compute_background_edge_density

    border_px  = kwargs.get("border_px", 60)
    min_radius = kwargs.get("min_radius", 25)
    max_radius = kwargs.get("max_radius", 250)

    bg_edge_density = compute_background_edge_density(image, border_px=border_px)
    hough = _hough_candidates(
        image, min_radius=min_radius, max_radius=max_radius,
        bg_edge_density=bg_edge_density,
    )
    filtered, edges = filter_false_positives(
        image, hough, bg_edge_density=bg_edge_density,
    ) if hough else ([], None)
    nms    = non_maximum_suppression(filtered, overlap_thresh=0.4) if filtered else []
    merged = _merge_by_arc_coverage(nms, edges) if (nms and edges is not None) else nms
    merged = _remove_cluster_wrappers(merged, edges)
    final  = _remove_nested_circles(merged, proximity_ratio=0.60) if merged else []

    return {
        "hough":           hough,
        "filtered":        filtered,
        "final":           final,
        "bg_edge_density": bg_edge_density,
    }
"""
detect.py — Coin Detection Pipeline

Hough circle detection with adaptive parameter selection, a four-stage
false-positive filter, and multi-step deduplication via NMS, arc-coverage
merge, cluster-wrapper removal, and nested-circle removal.
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# NMS and geometric post-processing
# ──────────────────────────────────────────────────────────────────────────────

def non_maximum_suppression(circles: list, overlap_thresh: float = 0.4) -> list:
    """Sort ascending by radius; suppress a candidate if it overlaps a kept
    circle by more than overlap_thresh of the summed radii.  Size-ratio guard
    0.75 prevents a small bimetallic inner ring from suppressing its outer coin."""
    if not circles:
        return []
    circles = sorted(circles, key=lambda c: c[2])
    keep = []
    for (x1, y1, r1) in circles:
        suppressed = any(
            np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < (r1 + r2) * (1.0 - overlap_thresh)
            and r2 >= r1 * 0.75
            for (x2, y2, r2) in keep
        )
        if not suppressed:
            keep.append((x1, y1, r1))
    return keep


def _circle_overlap_frac(x1, y1, r1, x2, y2, r2) -> float:
    """Fraction of the smaller circle's area covered by the intersection."""
    d = float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
    r_s, r_l = min(r1, r2), max(r1, r2)
    if d >= r_s + r_l:
        return 0.0
    if d <= r_l - r_s:
        return 1.0
    alpha = 2.0 * np.arccos(np.clip((d**2 + r_l**2 - r_s**2) / (2.0 * d * r_l), -1, 1))
    beta  = 2.0 * np.arccos(np.clip((d**2 + r_s**2 - r_l**2) / (2.0 * d * r_s), -1, 1))
    area  = 0.5 * r_l**2 * (alpha - np.sin(alpha)) + 0.5 * r_s**2 * (beta - np.sin(beta))
    return float(area / (np.pi * r_s**2))


def _merge_by_arc_coverage(circles: list, edges: np.ndarray,
                            overlap_thresh: float = 0.40,
                            ratio_range: tuple = (0.70, 1.30)) -> list:
    """Post-NMS: when two similar-size circles overlap >overlap_thresh of the
    smaller area, keep whichever has higher arc-coverage and discard the other."""
    if len(circles) <= 1:
        return circles
    result = list(circles)
    changed = True
    while changed:
        changed = False
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                x1, y1, r1 = result[i]
                x2, y2, r2 = result[j]
                r_ratio = min(r1, r2) / max(r1, r2) if max(r1, r2) > 0 else 0
                if not (ratio_range[0] <= r_ratio <= ratio_range[1]):
                    continue
                if _circle_overlap_frac(x1, y1, r1, x2, y2, r2) < overlap_thresh:
                    continue
                cov1 = _arc_coverage(edges, x1, y1, r1)
                cov2 = _arc_coverage(edges, x2, y2, r2)
                keep_idx = i if cov1 >= cov2 else j
                drop_idx = j if keep_idx == i else i
                result[i] = result[keep_idx]
                result.pop(drop_idx)
                changed = True
                break
            if changed:
                break
    return result


def _remove_nested_circles(circles: list, proximity_ratio: float = 0.60) -> list:
    """Remove a circle that is nested inside a larger one.

    Three conditions (any one triggers removal):
      1. Centre proximity  — small circle's centre is within proximity_ratio × r_large.
      2. Geometric containment — dist + r_small < r_large × 1.05 (fully inside).
      3. High overlap — the small circle's area is > 70 % covered by the large circle.
         Catches edge-sitting FPs that are mostly inside a real coin but not fully so.
    """
    if len(circles) <= 1:
        return circles
    result = []
    for (x1, y1, r1) in circles:
        nested = False
        for (x2, y2, r2) in circles:
            if r2 <= r1:
                continue
            d = float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
            if (d < r2 * proximity_ratio                  # centre proximity
                    or d + r1 < r2 * 1.05                 # geometric containment
                    or _circle_overlap_frac(x1, y1, r1, x2, y2, r2) > 0.70):  # mostly inside
                nested = True
                break
        if not nested:
            result.append((x1, y1, r1))
    return result


def _remove_cluster_wrappers(circles: list, edges: np.ndarray = None) -> list:
    """Remove circles that contain 2+ clearly smaller circles (r_enc < 0.75×r)
    within 0.85×r of their centre — these are over-sized Hough artefacts.

    Two guards prevent removing a real (partially-occluded) coin:
      1. Arc-coverage guard: if the candidate's own arc coverage >= 0.38 it has
         a real circular edge and should not be discarded as a wrapper.
      2. Enclosed-overlap guard: if the 2 enclosed circles overlap each other by
         >30 % of the smaller area, they are both FPs from the same coin region —
         the enclosing circle is the real detection, so keep it.
    """
    if len(circles) <= 2:
        return circles
    result = []
    for (x1, y1, r1) in circles:
        inside = [
            (x2, y2, r2) for (x2, y2, r2) in circles
            if (x2, y2, r2) != (x1, y1, r1)
            and np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < r1 * 0.85
            and r2 < r1 * 0.75
        ]
        if len(inside) < 2:
            result.append((x1, y1, r1))
            continue

        # Guard 1 — real coin edge: arc coverage of the candidate itself
        if edges is not None and _arc_coverage(edges, x1, y1, r1) >= 0.38:
            result.append((x1, y1, r1))
            continue

        # Guard 2 — enclosed FPs: the two enclosed circles overlap each other
        if len(inside) >= 2:
            xa, ya, ra = inside[0]
            xb, yb, rb = inside[1]
            if _circle_overlap_frac(xa, ya, ra, xb, yb, rb) > 0.30:
                result.append((x1, y1, r1))
                continue

        # Confirmed wrapper artefact — drop it
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Hough circle detection
# ──────────────────────────────────────────────────────────────────────────────

def _hough_candidates(
    image: np.ndarray,
    min_radius: int = 30,
    max_radius: int = 250,
    bg_edge_density: float = 0.0,
) -> list:
    """Ensemble of HOUGH_GRADIENT and HOUGH_GRADIENT_ALT.

    GRADIENT_ALT uses a quality threshold in [0,1] and produces fewer, more
    precise circles — primary source.  HOUGH_GRADIENT with adaptive param2
    is kept as secondary to catch coins missed by ALT.
    """
    from src.preprocess import preprocess, _scale_for_processing

    small, scale = _scale_for_processing(image)
    smoothed = preprocess(small)
    min_r    = max(1, int(min_radius * scale))
    max_r    = max(min_r + 1, int(max_radius * scale))
    min_dist = max(1, int(60 * scale))

    all_raw = []

    # Primary: HOUGH_GRADIENT_ALT
    # param2 quality: 0.90 (plain) → 0.75 (textured)
    param2_alt = float(np.clip(0.90 - 0.15 * bg_edge_density / 0.20, 0.75, 0.90))
    circles_alt = cv2.HoughCircles(
        smoothed, cv2.HOUGH_GRADIENT_ALT, dp=1.5,
        minDist=min_dist, param1=300, param2=param2_alt,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles_alt is not None:
        all_raw += [(int(round(x / scale)), int(round(y / scale)), int(round(r / scale)))
                    for (x, y, r) in np.around(circles_alt[0]).astype(np.int32).tolist()]

    # Secondary: HOUGH_GRADIENT (only when ALT returns very few candidates)
    if len(all_raw) < 3:
        param2_grad = int(np.clip(45 + 20 * bg_edge_density / 0.20, 45, 65))
        circles_grad = cv2.HoughCircles(
            smoothed, cv2.HOUGH_GRADIENT, dp=1,
            minDist=min_dist, param1=40, param2=param2_grad,
            minRadius=min_r, maxRadius=max_r,
        )
        if circles_grad is not None:
            all_raw += [(int(round(x / scale)), int(round(y / scale)), int(round(r / scale)))
                        for (x, y, r) in np.around(circles_grad[0]).astype(np.int32).tolist()]

    return non_maximum_suppression(all_raw, overlap_thresh=0.5)


# ──────────────────────────────────────────────────────────────────────────────
# False-positive filter
# ──────────────────────────────────────────────────────────────────────────────

_N72 = 72
_A72 = np.linspace(0, 2 * np.pi, _N72, endpoint=False)
_C72 = np.cos(_A72)
_S72 = np.sin(_A72)


def _arc_coverage(edges, x, y, r, n_sectors=36, ring_width=8):
    h, w = edges.shape[:2]
    sectors_hit = np.zeros(n_sectors, dtype=bool)
    r_inner = max(r - ring_width, 1);  r_outer = r + ring_width
    y0 = max(0, y - r_outer);  y1 = min(h, y + r_outer + 1)
    x0 = max(0, x - r_outer);  x1 = min(w, x + r_outer + 1)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dist_sq = (ys - y) ** 2 + (xs - x) ** 2
    in_ring = (dist_sq >= r_inner ** 2) & (dist_sq <= r_outer ** 2)
    eyx = np.argwhere(in_ring & (edges[y0:y1, x0:x1] > 0))
    for ey, ex in eyx:
        angle = np.arctan2(ey - (y - y0), ex - (x - x0))
        sectors_hit[int((angle + np.pi) / (2 * np.pi) * n_sectors) % n_sectors] = True
    return float(sectors_hit.sum()) / n_sectors


def _edge_density_on_ring(edges, x, y, r, ring_width=8):
    h, w = edges.shape[:2]
    mo = np.zeros((h, w), np.uint8);  mi = np.zeros((h, w), np.uint8)
    cv2.circle(mo, (x, y), r + ring_width, 255, -1)
    cv2.circle(mi, (x, y), max(r - ring_width, 0), 255, -1)
    ring = cv2.subtract(mo, mi)
    ring_px = np.count_nonzero(ring)
    return np.count_nonzero(cv2.bitwise_and(edges, edges, mask=ring)) / ring_px if ring_px else 0.0


def _color_uniformity_inside(image, x, y, r):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (x, y), max(int(r * 0.80), 1), 255, -1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat_mask = (hsv[:, :, 1] > 12).astype(np.uint8) * 255
    hue_vals = hsv[:, :, 0][cv2.bitwise_and(mask, sat_mask) > 0]
    if len(hue_vals) >= 10:
        return 1.0 / (1.0 + float(np.std(hue_vals.astype(float))) / 30.0)
    v_vals = hsv[:, :, 2][mask > 0].astype(float)
    return 1.0 / (1.0 + float(np.std(v_vals)) / 40.0) if len(v_vals) > 0 else 0.0


def _edge_density_at_radius(edges, cx, cy, sample_r):
    H, W = edges.shape
    xs = np.clip(np.round(cx + sample_r * _C72).astype(np.int32), 0, W - 1)
    ys = np.clip(np.round(cy + sample_r * _S72).astype(np.int32), 0, H - 1)
    return float((edges[ys, xs] > 0).mean())


def _ring_edge_sharpness(edges, cx, cy, r):
    ring  = _edge_density_at_radius(edges, cx, cy, r)
    inner = _edge_density_at_radius(edges, cx, cy, max(r * 0.55, 1.0))
    outer = _edge_density_at_radius(edges, cx, cy, r * 1.55)
    bg = (inner + outer) * 0.5
    return ring / bg if bg > 1e-4 else ring * 100.0


def _adaptive_sharpness_threshold(bg_edge_density: float,
                                   base: float = 1.2) -> float:
    """Smooth: 1.0 + (base-1.0)×exp(−10×bg_tex). Plain=base, textured→1.0."""
    return 1.0 + (base - 1.0) * float(np.exp(-10.0 * bg_edge_density))


def filter_false_positives(
    image: np.ndarray,
    circles: list,
    edge_density_min: float     = 0.08,
    color_uniformity_min: float = 0.28,
    arc_coverage_min: float     = 0.35,
    ring_sharpness_min: float   = 1.5,
    bg_edge_density: float      = None,
) -> tuple:
    """Four-stage FP filter. Returns (valid_circles, edges)."""
    from src.preprocess import build_enhanced_edges

    if image.ndim < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    edges = build_enhanced_edges(image)

    if bg_edge_density is not None:
        ring_sharpness_min = _adaptive_sharpness_threshold(bg_edge_density,
                                                            ring_sharpness_min)

    valid = []
    for (x, y, r) in circles:
        x, y, r = int(x), int(y), int(r)
        arc_cov = _arc_coverage(edges, x, y, r)
        if arc_cov                                    < arc_coverage_min:      continue
        if _edge_density_on_ring(edges, x, y, r)     < edge_density_min:      continue
        if _color_uniformity_inside(image, x, y, r)  < color_uniformity_min:  continue
        if _ring_edge_sharpness(edges, x, y, r)      < ring_sharpness_min:    continue
        valid.append((x, y, r))

    return valid, edges


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def detect_coins(
    image: np.ndarray,
    border_px: int  = 60,
    min_radius: int = 25,
    max_radius: int = 250,
) -> list:
    """Detect coins. Returns list of (x, y, radius) tuples."""
    from src.preprocess import compute_background_edge_density

    bg_edge_density = compute_background_edge_density(image, border_px=border_px)
    candidates = _hough_candidates(
        image, min_radius=min_radius, max_radius=max_radius,
        bg_edge_density=bg_edge_density,
    )
    if not candidates:
        return []

    filtered, edges = filter_false_positives(
        image, candidates, bg_edge_density=bg_edge_density,
    )
    if not filtered:
        return []

    deduped = non_maximum_suppression(filtered, overlap_thresh=0.4)
    merged  = _merge_by_arc_coverage(deduped, edges)
    merged  = _remove_cluster_wrappers(merged, edges)
    return _remove_nested_circles(merged, proximity_ratio=0.60)


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_debug_candidates(image: np.ndarray, **kwargs) -> dict:
    from src.preprocess import compute_background_edge_density

    border_px  = kwargs.get("border_px", 60)
    min_radius = kwargs.get("min_radius", 25)
    max_radius = kwargs.get("max_radius", 250)

    bg_edge_density = compute_background_edge_density(image, border_px=border_px)
    hough = _hough_candidates(
        image, min_radius=min_radius, max_radius=max_radius,
        bg_edge_density=bg_edge_density,
    )
    filtered, edges = filter_false_positives(
        image, hough, bg_edge_density=bg_edge_density,
    ) if hough else ([], None)
    nms    = non_maximum_suppression(filtered, overlap_thresh=0.4) if filtered else []
    merged = _merge_by_arc_coverage(nms, edges) if (nms and edges is not None) else nms
    merged = _remove_cluster_wrappers(merged, edges)
    final  = _remove_nested_circles(merged, proximity_ratio=0.60) if merged else []

    return {
        "hough":           hough,
        "filtered":        filtered,
        "final":           final,
        "bg_edge_density": bg_edge_density,
    }