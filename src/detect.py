"""
detect.py — Coin Detection Pipeline

Dual pipeline: color segmentation (foreground mask → contours → circles) and
Hough circles run independently; candidates are merged, FP-filtered, and
deduplicated via NMS + overlap merge + nested/wrapper removal.
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
    """Remove a circle whose centre lies within proximity_ratio × r_large of a
    larger circle's centre.  Single threshold for all radius ratios."""
    if len(circles) <= 1:
        return circles
    result = []
    for (x1, y1, r1) in circles:
        nested = any(
            r2 > r1
            and np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < r2 * proximity_ratio
            for (x2, y2, r2) in circles
        )
        if not nested:
            result.append((x1, y1, r1))
    return result


def _remove_cluster_wrappers(circles: list) -> list:
    """Remove circles that contain 2+ clearly smaller circles (r_enc < 0.75×r)
    within 0.85×r of their centre — these are merged-blob artefacts, not coins."""
    if len(circles) <= 2:
        return circles
    result = []
    for (x1, y1, r1) in circles:
        enclosed = sum(
            1 for (x2, y2, r2) in circles
            if (x2, y2, r2) != (x1, y1, r1)
            and np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < r1 * 0.85
            and r2 < r1 * 0.75
        )
        if enclosed < 2:
            result.append((x1, y1, r1))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline A: Color segmentation
# ──────────────────────────────────────────────────────────────────────────────

def _estimate_background_modes(lab: np.ndarray, border_px: int = 60,
                                n_modes: int = 3) -> np.ndarray:
    """K-means background model from border strips + low-gradient interior pixels."""
    h, w = lab.shape[:2]
    bp = min(border_px, h // 4, w // 4)
    border = np.vstack([
        lab[:bp, :].reshape(-1, 3),
        lab[max(0, h - bp):, :].reshape(-1, 3),
        lab[:, :bp].reshape(-1, 3),
        lab[:, max(0, w - bp):].reshape(-1, 3),
    ]).astype(np.float32)

    # Interior low-gradient pixels supplement the border model for complex
    # backgrounds (wood, burlap) where the centre of the image differs from
    # the border strips.  Skipped for uniform/plain backgrounds: their border
    # model is already accurate, and sampling coin-interior pixels (which also
    # have low gradient) would corrupt the background modes.
    border_complexity = float(np.std(border))
    if h > 2 * bp and w > 2 * bp and border_complexity > 12.0:
        l_chan = lab[bp:h - bp, bp:w - bp, 0].astype(np.float32)
        gy, gx = np.gradient(l_chan)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        low_mask = grad_mag < np.percentile(grad_mag, 25)
        interior = lab[bp:h - bp, bp:w - bp][low_mask].reshape(-1, 3).astype(np.float32)
        if len(interior) > 200:
            rng = np.random.RandomState(0)
            idx = rng.choice(len(interior), min(len(interior), 1000), replace=False)
            border = np.vstack([border, interior[idx]])

    n_modes = min(n_modes, len(border))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(
        border, n_modes, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    return centers


def _build_foreground_mask(
    image: np.ndarray,
    diff_thresh: float = 28.0,
    border_px: int = 60,
    n_bg_modes: int = 3,
) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_modes = _estimate_background_modes(lab, border_px=border_px, n_modes=n_bg_modes)
    dists = np.stack(
        [np.linalg.norm(lab - mode, axis=2) for mode in bg_modes], axis=0
    )
    mask = (np.min(dists, axis=0) > diff_thresh).astype(np.uint8) * 255
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open,  iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=3)
    return mask


def _watershed_split_blobs(mask: np.ndarray,
                            min_blob_area: int = 8000,
                            min_radius: int = 25) -> np.ndarray:
    """Split non-circular foreground blobs via distance-transform watershed."""
    result = mask.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_blob_area:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri > 0 and 4 * np.pi * area / (peri ** 2) >= 0.45:
            continue

        x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
        pad = 20
        x0 = max(x_b - pad, 0);  y0 = max(y_b - pad, 0)
        x1 = min(x_b + w_b + pad, mask.shape[1])
        y1 = min(y_b + h_b + pad, mask.shape[0])
        roi = mask[y0:y1, x0:x1].copy()

        dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        ks = max(3, min_radius // 3) | 1
        dilated = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
        peaks = ((dist == dilated) & (dist > 0.55)).astype(np.uint8)
        n_peaks, markers = cv2.connectedComponents(peaks)
        if n_peaks <= 2:
            continue

        markers_ws = markers.astype(np.int32)
        markers_ws[roi == 0] = -1
        cv2.watershed(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), markers_ws)

        new_roi = np.zeros_like(roi)
        min_area_px = 2.0 * np.pi * min_radius ** 2
        for label in range(1, markers_ws.max() + 1):
            if np.count_nonzero(markers_ws == label) > min_area_px:
                new_roi[markers_ws == label] = 255
        result[y0:y1, x0:x1] = new_roi
    return result


def _fit_circles_from_contours(
    contours,
    min_radius: int = 25,
    max_radius: int = 250,
    circularity_min: float = 0.60,
    solidity_min: float = 0.70,
) -> list:
    coins = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area == 0 or area / hull_area < solidity_min:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri > 0 and 4 * np.pi * area / (peri ** 2) < 0.50:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if not (min_radius <= radius <= max_radius):
            continue
        if area / (np.pi * radius ** 2) >= circularity_min:
            coins.append((int(cx), int(cy), int(radius)))
    return coins


def _color_segmentation_candidates(
    image: np.ndarray,
    diff_thresh: float = 24.0,
    border_px: int = 60,
    min_radius: int = 25,
    max_radius: int = 250,
    circularity_min: float = 0.60,
    _precomputed_mask: np.ndarray = None,
) -> list:
    mask = (_precomputed_mask if _precomputed_mask is not None
            else _build_foreground_mask(image, diff_thresh=diff_thresh, border_px=border_px))
    split = _watershed_split_blobs(mask, min_radius=min_radius)
    contours, _ = cv2.findContours(split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    candidates = _fit_circles_from_contours(
        contours, min_radius=min_radius, max_radius=max_radius,
        circularity_min=circularity_min,
    )
    return non_maximum_suppression(candidates, overlap_thresh=0.4)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline B: Hough circles (GRADIENT + GRADIENT_ALT ensemble)
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


def _mean_brightness_inside(image, x, y, r):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (x, y), max(int(r * 0.75), 1), 255, -1)
    vals = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2][mask > 0]
    return float(np.mean(vals)) if len(vals) > 0 else 0.0


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


def _foreground_fill_ratio(fg_mask, x, y, r):
    h, w = fg_mask.shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (x, y), max(int(r * 0.50), 1), 255, -1)
    px = np.count_nonzero(m)
    return np.count_nonzero(cv2.bitwise_and(fg_mask, fg_mask, mask=m)) / px if px else 0.0


def _rim_fill_ratio(fg_mask, x, y, r):
    h, w = fg_mask.shape[:2]
    mo = np.zeros((h, w), np.uint8);  mi = np.zeros((h, w), np.uint8)
    cv2.circle(mo, (x, y), max(int(r * 0.95), 1), 255, -1)
    cv2.circle(mi, (x, y), max(int(r * 0.75), 1), 255, -1)
    ann = cv2.subtract(mo, mi)
    ann_px = np.count_nonzero(ann)
    return np.count_nonzero(cv2.bitwise_and(fg_mask, fg_mask, mask=ann)) / ann_px if ann_px else 0.0


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
    brightness_min: float       = 50.0,
    ring_sharpness_min: float   = 1.5,
    fg_fill_min: float          = 0.12,
    fg_rim_fill_min: float      = 0.35,
    bg_edge_density: float      = None,
) -> tuple:
    """Six-stage FP filter. Returns (valid_circles, edges)."""
    from src.preprocess import build_enhanced_edges

    if image.ndim < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    edges = build_enhanced_edges(image)

    if bg_edge_density is not None:
        ring_sharpness_min = _adaptive_sharpness_threshold(bg_edge_density,
                                                            ring_sharpness_min)

    fill_mask = _build_foreground_mask(image, diff_thresh=20.0)

    valid = []
    for (x, y, r) in circles:
        x, y, r = int(x), int(y), int(r)
        arc_cov = _arc_coverage(edges, x, y, r)
        if arc_cov                                    < arc_coverage_min:      continue
        if _edge_density_on_ring(edges, x, y, r)     < edge_density_min:      continue
        if _color_uniformity_inside(image, x, y, r)  < color_uniformity_min:  continue
        if _mean_brightness_inside(image, x, y, r)   < brightness_min:        continue
        if _ring_edge_sharpness(edges, x, y, r)      < ring_sharpness_min:    continue
        # Fill check: overlapping coins can have low foreground fill at their
        # centres because the mask sees the merged cluster.  Skip the strict
        # fill thresholds when the edge arc is strongly confirmed (>=80 %).
        # But still require a non-trivial fill floor (>= 0.03) to reject
        # background texture circles that have zero actual foreground content.
        ff = _foreground_fill_ratio(fill_mask, x, y, r)
        rf = _rim_fill_ratio(fill_mask, x, y, r)
        if arc_cov >= 0.80:
            if ff < 0.03 and rf < 0.10:
                continue
        else:
            if ff < fg_fill_min and rf < fg_rim_fill_min:
                continue
        valid.append((x, y, r))

    return valid, edges


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def detect_coins(
    image: np.ndarray,
    diff_thresh: float     = 24.0,
    border_px: int         = 60,
    min_radius: int        = 25,
    max_radius: int        = 250,
    circularity_min: float = 0.60,
) -> list:
    """Detect coins. Returns list of (x, y, radius) tuples."""
    from src.preprocess import compute_background_edge_density

    bg_edge_density = compute_background_edge_density(image, border_px=border_px)
    fg_mask = _build_foreground_mask(image, diff_thresh=diff_thresh, border_px=border_px)

    color_candidates = _color_segmentation_candidates(
        image, diff_thresh=diff_thresh, border_px=border_px,
        min_radius=min_radius, max_radius=max_radius,
        circularity_min=circularity_min, _precomputed_mask=fg_mask,
    )
    hough_candidates = _hough_candidates(
        image, min_radius=min_radius, max_radius=max_radius,
        bg_edge_density=bg_edge_density,
    )

    all_candidates = color_candidates + hough_candidates
    if not all_candidates:
        return []

    filtered, edges = filter_false_positives(
        image, all_candidates, bg_edge_density=bg_edge_density,
    )
    if not filtered:
        return []

    deduped = non_maximum_suppression(filtered, overlap_thresh=0.4)
    merged  = _merge_by_arc_coverage(deduped, edges)
    merged  = _remove_cluster_wrappers(merged)
    return _remove_nested_circles(merged, proximity_ratio=0.60)


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_debug_mask(image: np.ndarray, diff_thresh: float = 24.0) -> np.ndarray:
    return cv2.cvtColor(_build_foreground_mask(image, diff_thresh=diff_thresh),
                        cv2.COLOR_GRAY2BGR)


def get_debug_candidates(image: np.ndarray, **kwargs) -> dict:
    from src.preprocess import compute_background_edge_density

    diff_thresh     = kwargs.get("diff_thresh", 24.0)
    border_px       = kwargs.get("border_px", 60)
    min_radius      = kwargs.get("min_radius", 25)
    max_radius      = kwargs.get("max_radius", 250)
    circularity_min = kwargs.get("circularity_min", 0.60)

    bg_edge_density = compute_background_edge_density(image, border_px=border_px)
    fg_mask = _build_foreground_mask(image, diff_thresh=diff_thresh, border_px=border_px)

    color = _color_segmentation_candidates(
        image, diff_thresh=diff_thresh, border_px=border_px,
        min_radius=min_radius, max_radius=max_radius,
        circularity_min=circularity_min, _precomputed_mask=fg_mask,
    )
    hough = _hough_candidates(
        image, min_radius=min_radius, max_radius=max_radius,
        bg_edge_density=bg_edge_density,
    )
    merged_raw = color + hough
    filtered, edges = filter_false_positives(image, merged_raw, bg_edge_density=bg_edge_density) \
        if merged_raw else ([], None)
    nms    = non_maximum_suppression(filtered, overlap_thresh=0.4) if filtered else []
    merged = _merge_by_arc_coverage(nms, edges) if (nms and edges is not None) else nms
    merged = _remove_cluster_wrappers(merged)
    final  = _remove_nested_circles(merged, proximity_ratio=0.60) if merged else []

    return {
        "color":           color,
        "hough":           hough,
        "filtered":        filtered,
        "final":           final,
        "bg_edge_density": bg_edge_density,
    }
