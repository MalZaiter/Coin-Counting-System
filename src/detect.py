"""
detect.py — Merged Coin Detection Pipeline

Strategy
--------
Both sub-pipelines run in parallel on every image, producing independent
candidate sets that are then combined and cleaned together:

  1. COLOR SEGMENTATION (primary, catches most coins)
       LAB background estimation → per-pixel distance mask → contour fitting
       Strong on plain/neutral backgrounds; fast; robust to partial overlap.

  2. HOUGH CIRCLES (fallback, catches what color segmentation misses)
       CLAHE + bilateral → HoughCircles → NMS
       Catches coins whose color is close to the background (image 040 case)
       and metallic coins under unusual lighting.

  3. MERGE + FALSE-POSITIVE FILTER (shared gate)
       Candidates from both pipelines are pooled, then run through the
       five-stage filter (arc coverage, edge density, colour uniformity,
       brightness, ring sharpness).  This is what rejects the false positives
       that color-only segmentation creates on textured backgrounds (image 044).

  4. GLOBAL NMS
       Final deduplication so the two pipelines don't double-count a coin.

Pipeline diagram
----------------
  image ──┬── color_segmentation() ──┐
          │                          ├── combine ──── filter_false_positives() ── nms() ── coins
          └── hough_pipeline()  ──┘
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Shared: Non-maximum suppression
# ──────────────────────────────────────────────────────────────────────────────

def non_maximum_suppression(circles: list, overlap_thresh: float = 0.4) -> list:
    """
    Remove overlapping detections that correspond to the same coin.

    Two circles are considered duplicates if their centre distance is less than
    (1 - overlap_thresh) × the sum of their radii.

    Sort order: ASCENDING radius (smallest kept first).

    Why ascending?
    --------------
    The color-segmentation pipeline sometimes produces one large circle that
    encloses an entire cluster of touching coins (e.g. three coins arranged
    in a triangle).  The Hough pipeline, meanwhile, correctly finds the
    individual coin circles.  With descending sort the large enclosing circle
    would be added first and then suppress all the correct small circles.

    With ascending sort the small individual coin circles are locked in first;
    the large cluster circle is then tested and suppressed because its centre
    coincides with one of the already-kept coins (distance < threshold).

    For same-coin duplicates from both pipelines (similar radii, nearby
    centres) the smaller of the two is kept — acceptable since both are
    good fits and the size difference is typically < 5 %.
    """
    if len(circles) == 0:
        return []
    circles = sorted(circles, key=lambda c: c[2])   # ascending: keep smaller
    keep = []
    for (x1, y1, r1) in circles:
        suppressed = any(
            np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < (r1 + r2) * (1.0 - overlap_thresh)
            for (x2, y2, r2) in keep
        )
        if not suppressed:
            keep.append((x1, y1, r1))
    return keep


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline A: Color segmentation
# ──────────────────────────────────────────────────────────────────────────────

def _estimate_background_modes(lab: np.ndarray, border_px: int = 60,
                                n_modes: int = 3) -> np.ndarray:
    """
    Estimate multiple background color modes from the image border strip.

    A single median fails on multicolored backgrounds such as peeling painted
    wood (teal + white paint stripes + pink/red grain).  K-means on the border
    pixels finds up to n_modes distinct background colors; a pixel is only
    classified as foreground if it is far from ALL modes.

    n_modes=3 works well in practice:
      - Uniform backgrounds: all 3 centers converge to the same color (no harm).
      - Bi-color backgrounds (wood grain): 2-3 centers cover the variation.
      - 3-color peeling paint (011/032/036/057): each center maps to one color.
    """
    border = np.vstack([
        lab[:border_px, :].reshape(-1, 3),
        lab[-border_px:, :].reshape(-1, 3),
        lab[:, :border_px].reshape(-1, 3),
        lab[:, -border_px:].reshape(-1, 3),
    ]).astype(np.float32)

    n_modes = min(n_modes, len(border))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(
        border, n_modes, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS
    )
    return centers  # shape: (n_modes, 3)


def _build_foreground_mask(
    image: np.ndarray,
    diff_thresh: float = 28.0,
    border_px: int = 60,
    n_bg_modes: int = 3,
) -> np.ndarray:
    """
    Binary mask of pixels significantly different from ALL background modes.

    LAB color space is used because Euclidean distance in LAB corresponds to
    perceptual color difference.  diff_thresh=28 ≈ "two visibly distinct paint
    colours apart."

    Multi-modal background
    ----------------------
    For each pixel, we compute the distance to the NEAREST background mode
    (rather than a single median).  A pixel is foreground only if it is far
    from every mode.  This is critical for surfaces with multiple background
    colors (peeling paint, printed fabric, wood grain).

    Morphological pipeline
    ----------------------
    OPEN  (5px ×2): removes isolated noise pixels and thin texture lines.
    CLOSE (11px ×3): fills glare patches and low-contrast inner coin regions.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_modes = _estimate_background_modes(lab, border_px=border_px, n_modes=n_bg_modes)

    dists = np.stack(
        [np.linalg.norm(lab - mode, axis=2) for mode in bg_modes], axis=0
    )
    min_dist = np.min(dists, axis=0)
    mask = (min_dist > diff_thresh).astype(np.uint8) * 255

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open,  iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=3)

    return mask


def _fit_circles_from_contours(
    contours,
    min_radius: int = 25,
    max_radius: int = 150,
    circularity_min: float = 0.60,
    solidity_min: float = 0.70,
) -> list:
    """
    Fit a minimum enclosing circle to each contour and keep circular ones.

    Circularity = contour_area / enclosing_circle_area
        1.0  → perfect circle
        0.65–0.95 → real coin (wear, partial overlap, glare patches)
        < 0.60 → elongated shadow or irregular blob

    Solidity = contour_area / convex_hull_area
        Rejects blobs formed by merging paint patches or printed fabric elements.
        These look large and have an irregular, non-convex perimeter even when
        their enclosing circle is reasonably circular.
        Real coins: solidity ≥ 0.80 (slight coin-edge scalloping → 0.80–0.99)
        Paint patch merges: solidity 0.20–0.60 (concave bays between patches)

    Isoperimetric circularity = 4π·area / perimeter²
        Rejects clusters of touching/overlapping coins (025, 095 case).
        A single coin has a smooth perimeter → iso_circ 0.65–0.90.
        Two or three touching coins produce a scalloped, bowed-in perimeter
        between each contact point → iso_circ drops to 0.25–0.50.
        Threshold 0.50 sits cleanly in the gap.

    max_radius=150 (default)
        At typical phone-camera distances, euro coin radii are 30–130 px.
        Anything larger is almost certainly a paint/texture blob, not a coin.
        Pass a larger value explicitly if shooting from very close range.
    """
    coins = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < solidity_min:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            iso_circularity = 4 * np.pi * area / (perimeter ** 2)
            if iso_circularity < 0.50:
                continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)

        if not (min_radius <= radius <= max_radius):
            continue

        circularity = area / (np.pi * radius ** 2)
        if circularity >= circularity_min:
            coins.append((int(cx), int(cy), int(radius)))

    return coins


def _color_segmentation_candidates(
    image: np.ndarray,
    diff_thresh: float = 24.0,
    border_px: int = 60,
    min_radius: int = 25,
    max_radius: int = 150,
    circularity_min: float = 0.60,
    _precomputed_mask: np.ndarray = None,
) -> list:
    """
    Run the full color-segmentation sub-pipeline and return raw candidates.

    Returns a list of (x, y, radius) tuples before any false-positive filtering.
    Accepts an optional _precomputed_mask to avoid redundant computation when
    the caller (detect_coins) has already built the foreground mask.
    """
    mask = (
        _precomputed_mask
        if _precomputed_mask is not None
        else _build_foreground_mask(image, diff_thresh=diff_thresh, border_px=border_px)
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []

    candidates = _fit_circles_from_contours(
        contours,
        min_radius=min_radius,
        max_radius=max_radius,
        circularity_min=circularity_min,
    )
    return non_maximum_suppression(candidates, overlap_thresh=0.4)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline B: Hough circles
# ──────────────────────────────────────────────────────────────────────────────

def _hough_candidates(
    image: np.ndarray,
    min_radius: int = 30,
    max_radius: int = 150,
) -> list:
    """
    Run the Hough-based sub-pipeline and return raw candidates.

    Uses preprocess() from preprocess.py to produce the smoothed grayscale
    that HoughCircles needs for correct gradient-direction voting.

    Returns a list of (x, y, radius) tuples before any false-positive filtering.

    HoughCircles parameter notes
    ----------------------------
    param2=45  (was 50, then 40): compromise between recall and precision.
               40 was too sensitive on dark fabric (040), producing many Hough
               candidates that overwhelmed the FP filter.  45 still improves
               recall over 50 for faint coins on similar-toned backgrounds
               while keeping false votes under control.
    minDist=60 (was 80): allows detecting closely spaced small coins without
               merging their peaks prematurely.
    """
    from src.preprocess import preprocess

    smoothed = preprocess(image)
    circles = cv2.HoughCircles(
        smoothed,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=60,
        param1=40,
        param2=45,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return []

    raw = np.uint16(np.around(circles[0])).tolist()
    return non_maximum_suppression(raw, overlap_thresh=0.5)


# ──────────────────────────────────────────────────────────────────────────────
# Shared: False-positive filter (five-stage)
# ──────────────────────────────────────────────────────────────────────────────

# Precomputed 72-point unit-circle sampling arrays
_N72 = 72
_A72 = np.linspace(0, 2 * np.pi, _N72, endpoint=False)
_C72 = np.cos(_A72)
_S72 = np.sin(_A72)


def _arc_coverage(edges: np.ndarray, x: int, y: int, r: int,
                  n_sectors: int = 36, ring_width: int = 8) -> float:
    """Fraction of the 360° arc that has edge support (sector voting)."""
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
    """Fraction of edge pixels on a thin ring at the detected circle boundary."""
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
    Color uniformity inside the detected circle.

    Primary: hue std over saturated pixels (saturation threshold 12 to capture
    desaturated copper/bronze).  Fallback: brightness std when too few
    saturated pixels exist (silver coins, specular highlights).
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
    """Fraction of 72 sample points at radius sample_r that land on an edge pixel."""
    H, W = edges.shape
    xs = np.clip(np.round(cx + sample_r * _C72).astype(np.int32), 0, W - 1)
    ys = np.clip(np.round(cy + sample_r * _S72).astype(np.int32), 0, H - 1)
    return float((edges[ys, xs] > 0).mean())


def _ring_edge_sharpness(edges: np.ndarray, cx: int, cy: int, r: int) -> float:
    """
    Ratio of edge density AT the coin boundary vs. inner/outer background.

    score = ring_density / mean(inner_density @ 0.55r, outer_density @ 1.55r)

    True coins:           strong ring, quiet interior/exterior → score >> 1
    Textured backgrounds: edges everywhere, ratio near 1.0 → low score

    Primary rejection filter for textured backgrounds that pass geometric and
    colour checks (wood, fabric, patterned carpet).
    Threshold 1.5 sits cleanly in the true/false-positive gap.
    """
    ring  = _edge_density_at_radius(edges, cx, cy, r)
    inner = _edge_density_at_radius(edges, cx, cy, max(r * 0.55, 1.0))
    outer = _edge_density_at_radius(edges, cx, cy, r * 1.55)
    bg = (inner + outer) * 0.5
    if bg < 1e-4:
        return ring * 100.0
    return ring / bg


def _foreground_fill_ratio(fg_mask: np.ndarray, x: int, y: int, r: int) -> float:
    """
    Fraction of the CENTER HALF of the circle that is foreground in the
    color-segmentation mask (i.e., visually distinct from the background).

    Why center rather than full circle
    ------------------------------------
    Evaluating at 85 % of r fails when a Hough circle is too large and
    accidentally encompasses real coins at its periphery: the coins contribute
    foreground pixels even though the circle is NOT centered on a coin.  A
    typical example is 040 (dark fabric) where Hough finds a circle centred on
    bare fabric between two coins; at 85 % r the coins at the edge of the circle
    push the fill above threshold.

    Using the INNER 50 % of r focuses strictly on what the detected circle is
    centred on.  A real coin → centre is the coin → fill ≥ 0.40.  A Hough
    circle centred on background (fabric gap, empty teal wood, bare concrete
    texture) → fill ≈ 0.0 even if coins happen to sit near the perimeter.

    The fill mask is built externally with diff_thresh = 20 (lower than the
    main pipeline's 24) so that silver/grey coins on near-matching backgrounds
    show up more readily.
    """
    h, w = fg_mask.shape[:2]
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (x, y), max(int(r * 0.50), 1), 255, -1)
    circle_pixels = np.count_nonzero(circle_mask)
    if circle_pixels == 0:
        return 0.0
    fg_pixels = np.count_nonzero(cv2.bitwise_and(fg_mask, fg_mask, mask=circle_mask))
    return fg_pixels / circle_pixels


def _rim_fill_ratio(fg_mask: np.ndarray, x: int, y: int, r: int) -> float:
    """
    Fill ratio in the OUTER RIM annulus (75 % to 95 % of radius).

    Purpose — bimetallic coin fallback
    ------------------------------------
    Euro 1 € and 2 € coins have a distinctive gold/nickel outer ring whose
    colour contrasts strongly with any background.  When the coin is face-up
    showing the silver centre, the inner-50%-radius fill check yields ≈ 0
    (silver ≈ white/grey background).  The outer rim zone, however, is gold
    or brass and registers clearly in the foreground mask even at diff_thresh
    = 20.

    Evaluating 75 %–95 % of r places the window squarely over the rim zone
    for typical bimetallic sizes (r 55–100 px), while staying tight enough to
    avoid the background just outside the coin edge.

    Why the threshold is higher (0.35 vs 0.12 for center fill)
    -----------------------------------------------------------
    For large Hough false-positives that encompass several real coins (040
    dark-fabric pattern), those coins may partially fall inside the 75-95 %
    annulus and produce a moderate fill score.  Requiring 0.35 means at least
    35 % of the rim annulus must be foreground — achievable for a solid gold
    rim (≈ 50–70 % fill) but unlikely for a circle whose perimeter merely
    clips a few scattered coins (≈ 10–25 % fill).
    """
    h, w = fg_mask.shape[:2]
    outer = np.zeros((h, w), dtype=np.uint8)
    inner = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(outer, (x, y), max(int(r * 0.95), 1), 255, -1)
    cv2.circle(inner, (x, y), max(int(r * 0.75), 1), 255, -1)
    annulus = cv2.subtract(outer, inner)
    annulus_px = np.count_nonzero(annulus)
    if annulus_px == 0:
        return 0.0
    fg_px = np.count_nonzero(cv2.bitwise_and(fg_mask, fg_mask, mask=annulus))
    return fg_px / annulus_px


def filter_false_positives(
    image: np.ndarray,
    circles: list,
    edge_density_min: float     = 0.08,
    color_uniformity_min: float = 0.25,
    arc_coverage_min: float     = 0.35,
    brightness_min: float       = 50.0,
    ring_sharpness_min: float   = 1.5,
    fg_fill_min: float          = 0.12,
    fg_rim_fill_min: float      = 0.35,
) -> list:
    """
    Six-stage false-positive filter applied to the merged candidate set.

    Filters (applied in order; early exit on first failure)
    --------------------------------------------------------
    1. Arc coverage      ≥ 0.35  — coin arcs span most of the perimeter.
    2. Edge density      ≥ 0.08  — strong edge concentration at the radius.
    3. Color uniformity  ≥ 0.25  — metallic interior vs textured background.
    4. Brightness        ≥ 50    — reject shadow-filled carpet indentations
                                   (carpet pits mean V ≈ 40, coins ≈ 80–180).
    5. Ring sharpness    ≥ 1.5   — edge density AT boundary / background.
                                   Lowered from 1.8 back to 1.5: granular
                                   surfaces (concrete) compress the ratio even
                                   for real coins; stage 6 takes over as the
                                   primary FP rejector.
    6. Fill gate (OR)    — the circle must pass at LEAST ONE of:
          a. Center fill ≥ 0.12  — ≥ 12 % of the inner 50 % radius is
                                   foreground.  Primary gate for mono-colour
                                   coins (copper, gold, dark silver).
          b. Rim fill    ≥ 0.35  — ≥ 35 % of the outer annulus (75–95 % of
                                   radius) is foreground.  Fallback for
                                   bimetallic coins (1 €, 2 €) whose silver
                                   centre resembles the background; their gold
                                   rim registers in the fill mask even when
                                   the centre does not.

       Both checks use a freshly-computed LAB mask with diff_thresh = 20,
       intentionally lower than the main pipeline (24) so that silver/grey
       surfaces are more readily classified as foreground.

       The center check at r × 0.50 rejects large Hough circles that are
       centred on background but encompass real coins at their periphery
       (040 dark-fabric pattern): the circle centre is fabric → fill ≈ 0,
       and the rim check also fails because the scattered coins only fill
       ≈ 10–25 % of the 75–95 % annulus.
    """
    from src.preprocess import build_enhanced_edges

    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    edges = build_enhanced_edges(image)

    fill_mask = None
    if fg_fill_min > 0 or fg_rim_fill_min > 0:
        fill_mask = _build_foreground_mask(image, diff_thresh=20.0)

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
        if fill_mask is not None:
            center_ok = _foreground_fill_ratio(fill_mask, x, y, r) >= fg_fill_min
            rim_ok    = _rim_fill_ratio(fill_mask, x, y, r) >= fg_rim_fill_min
            if not (center_ok or rim_ok):
                continue

        valid.append((x, y, r))

    return valid


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def detect_coins(
    image: np.ndarray,
    diff_thresh: float     = 24.0,
    border_px: int         = 60,
    min_radius: int        = 25,
    max_radius: int        = 150,
    circularity_min: float = 0.60,
) -> list:
    """
    Detect coins using the merged dual-pipeline strategy.

    Both sub-pipelines run unconditionally and their candidates are pooled
    before the shared false-positive filter runs.  This means:

    - Color segmentation catches coins that Hough misses (partial overlap,
      unusual angles, clustered coins).
    - Hough catches coins that color segmentation misses (coin color similar
      to background, heavy shadow, strong directional lighting).
    - The false-positive filter rejects noise from color segmentation on
      textured backgrounds (carpet, fabric, wood grain).

    The foreground mask is built once and shared with both the color-
    segmentation pipeline (contour source) and the false-positive filter
    (foreground fill check) to avoid redundant computation.

    Parameters
    ----------
    image           : BGR image (cv2.imread or camera frame)
    diff_thresh     : LAB distance threshold from background (default 24)
                      Lower → catch more coins on similar-toned backgrounds;
                      raise if too much background noise leaks through.
                      Lowered from 28 → 24 to recover gold coins on concrete
                      (024/043) whose LAB distance was borderline at 28.
    border_px       : Border strip width for background estimation
    min_radius      : Minimum expected coin radius in pixels
    max_radius      : Maximum expected coin radius in pixels
    circularity_min : Minimum area/enclosing-circle-area ratio (color path)

    Returns
    -------
    List of (x, y, radius) tuples, one per detected coin.

    Tuning tips
    -----------
    Coins missed      → lower diff_thresh (18–23) or lower circularity_min
    False positives   → raise diff_thresh (30+) or raise fg_fill_min
    Wrong sizes       → adjust min_radius / max_radius for your camera distance
    """
    # --- Build foreground mask once; share with sub-pipeline A and filter ---
    fg_mask = _build_foreground_mask(image, diff_thresh=diff_thresh, border_px=border_px)

    # --- Sub-pipeline A: color segmentation (uses pre-computed mask) ---
    color_candidates = _color_segmentation_candidates(
        image,
        diff_thresh=diff_thresh,
        border_px=border_px,
        min_radius=min_radius,
        max_radius=max_radius,
        circularity_min=circularity_min,
        _precomputed_mask=fg_mask,
    )

    # --- Sub-pipeline B: Hough circles ---
    hough_candidates = _hough_candidates(image, min_radius=min_radius, max_radius=max_radius)

    # --- Merge ---
    all_candidates = color_candidates + hough_candidates

    if len(all_candidates) == 0:
        return []

    # --- Shared false-positive filter ---
    filtered = filter_false_positives(image, all_candidates)

    if len(filtered) == 0:
        return []

    # --- Global NMS: deduplicate coins seen by both pipelines ---
    return non_maximum_suppression(filtered, overlap_thresh=0.4)


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_debug_mask(image: np.ndarray, diff_thresh: float = 24.0) -> np.ndarray:
    """
    Return the color-segmentation foreground mask as a BGR image for inspection.

    Usage:
        cv2.imwrite("debug_mask.jpg", get_debug_mask(img))

    White = foreground (coin candidates), black = background.
    If coins are missing: lower diff_thresh.
    If background noise appears: raise diff_thresh.
    """
    mask = _build_foreground_mask(image, diff_thresh=diff_thresh)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def get_debug_candidates(image: np.ndarray, **kwargs) -> dict:
    """
    Return both pipeline candidate sets and the filtered result for debugging.

    Returns a dict with keys:
      'color'    : raw candidates from color segmentation
      'hough'    : raw candidates from Hough pipeline
      'filtered' : candidates that passed the false-positive filter
      'final'    : final detections after global NMS
    """
    diff_thresh = kwargs.get("diff_thresh", 24.0)
    border_px   = kwargs.get("border_px", 60)
    min_radius  = kwargs.get("min_radius", 25)
    max_radius  = kwargs.get("max_radius", 150)

    fg_mask = _build_foreground_mask(image, diff_thresh=diff_thresh, border_px=border_px)

    color = _color_segmentation_candidates(
        image,
        diff_thresh=diff_thresh,
        border_px=border_px,
        min_radius=min_radius,
        max_radius=max_radius,
        circularity_min=kwargs.get("circularity_min", 0.60),
        _precomputed_mask=fg_mask,
    )
    hough = _hough_candidates(image, min_radius=min_radius, max_radius=max_radius)
    merged = color + hough
    filtered = filter_false_positives(image, merged) if merged else []
    final = non_maximum_suppression(filtered, overlap_thresh=0.4) if filtered else []

    return {
        "color":    color,
        "hough":    hough,
        "filtered": filtered,
        "final":    final,
    }
