"""
detect.py — Circle Detection & Filtering

Responsibilities:
- Detect circular objects using Hough Transform
- Remove false positives (e.g. bottle caps, background texture)

Pipeline:
1. preprocess()        → smoothed grayscale for HoughCircles
2. HoughCircles        → candidate circles
3. NMS                 → remove duplicate detections on the same coin
4. filter_false_positives → five-stage filter using the original image +
                            build_enhanced_edges() for arc/density checks
"""

import cv2
import numpy as np

# Precomputed 72-point unit circle sampling arrays (shared by ring sharpness filter)
_N72 = 72
_A72 = np.linspace(0, 2 * np.pi, _N72, endpoint=False)
_C72 = np.cos(_A72)
_S72 = np.sin(_A72)


def detect_circles(image: np.ndarray) -> list:
    """
    Detect circular objects in a smoothed grayscale image using HoughCircles.

    Input must be the smoothed grayscale from preprocess() — NOT a binary edge
    map.  HoughCircles (HOUGH_GRADIENT) computes gradient direction internally;
    feeding it a grayscale preserves accurate inward-pointing gradients at coin
    boundaries, so votes accumulate tightly at the correct (x, y, r) bin.

    Parameters
    ----------
    dp=1        Full-resolution accumulator for accurate circle placement.
    minDist=80  Minimum centre-to-centre distance.
    param1=40   Upper Canny threshold inside HoughCircles.
    param2=50   Minimum accumulator votes. Lower values find more candidates
                (including partial-arc coins on clean backgrounds); the ring
                sharpness filter rejects the false positives on textured
                backgrounds that a lower param2 would admit.
    minRadius=30 / maxRadius=220
                Bracket the expected euro coin pixel-radius range at typical
                phone-camera shooting distances.
    """
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=80,
        param1=40,
        param2=50,
        minRadius=30,
        maxRadius=220,
    )

    if circles is not None:
        return np.uint16(np.around(circles[0])).tolist()
    return []


def non_maximum_suppression(circles: list, overlap_thresh: float = 0.5) -> list:
    """
    Remove overlapping circle detections that represent the same coin.

    Suppression condition: if the distance between two centres is less than
    (1 - overlap_thresh) × the SUM of their radii, the smaller circle is
    suppressed.  overlap_thresh=0.5 allows closely spaced distinct coins while
    collapsing multiple detections on a single coin.
    """
    if len(circles) == 0:
        return []

    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    keep = []

    for (x1, y1, r1) in circles:
        suppressed = False
        for (x2, y2, r2) in keep:
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist < (r1 + r2) * (1.0 - overlap_thresh):
                suppressed = True
                break
        if not suppressed:
            keep.append((x1, y1, r1))

    return keep


def compute_circularity(contour) -> float:
    """Circularity = 4π·area / perimeter² (perfect circle = 1.0)."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)


def compute_solidity(contour) -> float:
    """Solidity = contour area / convex hull area."""
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return area / hull_area


def _arc_coverage(edges: np.ndarray, x: int, y: int, r: int,
                  n_sectors: int = 36, ring_width: int = 8) -> float:
    """
    Fraction of the circle's 360° arc that has edge support.

    Divide the perimeter into n_sectors angular wedges; a wedge "passes" if any
    edge pixel falls within it.  Real coins produce a near-complete arc;
    background noise produces a few scattered sectors.
    """
    h, w = edges.shape[:2]
    sectors_hit = np.zeros(n_sectors, dtype=bool)

    r_inner = max(r - ring_width, 1)
    r_outer = r + ring_width

    y_min = max(0, y - r_outer)
    y_max = min(h, y + r_outer + 1)
    x_min = max(0, x - r_outer)
    x_max = min(w, x + r_outer + 1)

    ys, xs = np.mgrid[y_min:y_max, x_min:x_max]
    dy = ys - y
    dx = xs - x
    dist_sq = dy ** 2 + dx ** 2

    in_ring = (dist_sq >= r_inner ** 2) & (dist_sq <= r_outer ** 2)
    ring_edges = edges[y_min:y_max, x_min:x_max]

    edge_ys, edge_xs = np.where(in_ring & (ring_edges > 0))

    for ey, ex in zip(edge_ys, edge_xs):
        angle = np.arctan2(ey - (y - y_min), ex - (x - x_min))
        sector = int((angle + np.pi) / (2 * np.pi) * n_sectors) % n_sectors
        sectors_hit[sector] = True

    return float(np.sum(sectors_hit)) / n_sectors


def _edge_density_on_ring(edges: np.ndarray, x: int, y: int, r: int,
                           ring_width: int = 8) -> float:
    """
    Fraction of edge pixels on a thin ring at the detected circle boundary.

    Real coins have a strong circular edge; background false positives do not
    concentrate edge pixels at a specific radius.
    """
    h, w = edges.shape[:2]
    mask_outer = np.zeros((h, w), dtype=np.uint8)
    mask_inner = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_outer, (x, y), r + ring_width, 255, -1)
    cv2.circle(mask_inner, (x, y), max(r - ring_width, 0), 255, -1)
    ring_mask = cv2.subtract(mask_outer, mask_inner)

    ring_pixels = np.count_nonzero(ring_mask)
    if ring_pixels == 0:
        return 0.0

    edge_pixels = np.count_nonzero(cv2.bitwise_and(edges, edges, mask=ring_mask))
    return edge_pixels / ring_pixels


def _color_uniformity_inside(image: np.ndarray, x: int, y: int, r: int) -> float:
    """
    Colour uniformity inside the detected circle.

    Primary path — hue uniformity over saturated pixels:
        Coins have a consistent metallic colour.  Saturation threshold is 12
        (not 25) so that desaturated copper/bronze coins qualify.

    Fallback — brightness uniformity:
        When fewer than 10 saturated pixels exist (silver coins, specular
        highlights, coins under unusual tinted lighting), measure V-channel
        std instead.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    inner_r = max(int(r * 0.80), 1)
    cv2.circle(mask, (x, y), inner_r, 255, -1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    sat_mask = (sat > 12).astype(np.uint8) * 255
    combined = cv2.bitwise_and(mask, sat_mask)
    hue_vals = hue[combined > 0]

    if len(hue_vals) >= 10:
        std_hue = float(np.std(hue_vals.astype(float)))
        return 1.0 / (1.0 + std_hue / 30.0)
    else:
        v_vals = val[mask > 0].astype(float)
        if len(v_vals) == 0:
            return 0.0
        std_v = float(np.std(v_vals))
        return 1.0 / (1.0 + std_v / 40.0)


def _mean_brightness_inside(image: np.ndarray, x: int, y: int, r: int) -> float:
    """Mean V-channel brightness inside the circle. Very dark regions are not coins."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    inner_r = max(int(r * 0.75), 1)
    cv2.circle(mask, (x, y), inner_r, 255, -1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    vals = hsv[:, :, 2][mask > 0]

    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))


def _edge_density_at_radius(edges: np.ndarray, cx: int, cy: int,
                             sample_r: float) -> float:
    """
    Fraction of the 72 equally-spaced sample points at radius sample_r that
    land on an edge pixel.  Used by _ring_edge_sharpness.
    """
    H, W = edges.shape
    xs = np.clip(np.round(cx + sample_r * _C72).astype(np.int32), 0, W - 1)
    ys = np.clip(np.round(cy + sample_r * _S72).astype(np.int32), 0, H - 1)
    return float((edges[ys, xs] > 0).mean())


def _ring_edge_sharpness(edges: np.ndarray, cx: int, cy: int, r: int) -> float:
    """
    Ratio of edge density AT the coin boundary vs. the local background.

    Compares three sampling rings:
        ring  — at radius r        (the coin edge itself)
        inner — at radius 0.55 r   (background inside the coin)
        outer — at radius 1.55 r   (background outside the coin)

    score = ring_density / mean(inner_density, outer_density)

    Real coins:           strong ring, quiet interior/exterior → score >> 1
                          (on plain backgrounds the bg ≈ 0, so we cap at 100×)
    Textured background:  edges everywhere, ratio near 1.0 → low score

    Typical values
    --------------
    True positives  : p10 = 1.84,  median = 28.8
    False positives : p90 = 1.43

    Threshold 1.5 sits cleanly in the gap (confirmed by diagnose_sharpness.py).
    """
    ring  = _edge_density_at_radius(edges, cx, cy, r)
    inner = _edge_density_at_radius(edges, cx, cy, max(r * 0.55, 1.0))
    outer = _edge_density_at_radius(edges, cx, cy, r * 1.55)
    bg = (inner + outer) * 0.5
    if bg < 1e-4:
        return ring * 100.0
    return ring / bg


def filter_false_positives(
    image: np.ndarray,
    circles: list,
    edge_density_min: float = 0.08,
    color_uniformity_min: float = 0.25,
    arc_coverage_min: float = 0.35,
    brightness_min: float = 30.0,
    ring_sharpness_min: float = 1.5,
) -> list:
    """
    Five-stage false-positive filter using the original image.

    Filters (applied in order, early exit on first failure)
    --------------------------------------------------------
    1. Arc coverage         ≥ 0.35 — coin arcs span most of the perimeter.
    2. Edge density         ≥ 0.08 — strong edge concentration at radius.
    3. Colour uniformity    ≥ 0.25 — metallic interior vs textured background.
    4. Brightness           ≥ 30   — reject shadows and holes.
    5. Ring edge sharpness  ≥ 1.5  — edge density AT the boundary must be ≥ 1.5×
                                     the mean of inner/outer background density.
                                     This is the primary rejection filter for
                                     textured backgrounds (wood, fabric, etc.)
                                     that pass all geometric/colour checks.
                                     Replaces gradient_consistency (Phase 2)
                                     which was fundamentally limited by the
                                     2/π ≈ 0.637 isotropic-texture ceiling.
    """
    from src.preprocess import build_enhanced_edges

    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    edges = build_enhanced_edges(image)

    valid = []
    for (x, y, r) in circles:
        x, y, r = int(x), int(y), int(r)

        if _arc_coverage(edges, x, y, r) < arc_coverage_min:
            continue

        if _edge_density_on_ring(edges, x, y, r) < edge_density_min:
            continue

        if _color_uniformity_inside(image, x, y, r) < color_uniformity_min:
            continue

        if _mean_brightness_inside(image, x, y, r) < brightness_min:
            continue

        if _ring_edge_sharpness(edges, x, y, r) < ring_sharpness_min:
            continue

        valid.append((x, y, r))

    return valid


def detect_coins(image: np.ndarray) -> list:
    """
    Full detection pipeline:
    1. preprocess()              → smoothed grayscale for HoughCircles
    2. detect_circles()          → Hough candidates
    3. non_maximum_suppression() → remove duplicates on the same coin
    4. filter_false_positives()  → arc + edge + colour + brightness +
                                   ring sharpness checks

    Returns:
        List of valid coin candidates as (x, y, radius) tuples.
    """
    from src.preprocess import preprocess

    smoothed = preprocess(image)
    circles  = detect_circles(smoothed)

    if len(circles) == 0:
        return []

    circles    = non_maximum_suppression(list(circles), overlap_thresh=0.5)
    valid_coins = filter_false_positives(image, circles)

    return valid_coins
