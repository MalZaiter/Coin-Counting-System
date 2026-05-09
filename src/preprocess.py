"""
preprocess.py — Image Preprocessing Utilities

Changes vs original
--------------------
1. build_enhanced_edges(): downscales to MAX_PROCESS_DIM before bilateralFilter(d=13),
   scales edges back up.  Eliminates 20-60s bottleneck on large images.

2. compute_background_edge_density(): NEW — measures border-strip edge density at
   FULL resolution (processes border strips only, not whole image).  Must be full-res
   because fine textile texture (denim, burlap) is destroyed by downscaling.

3. preprocess(): removed fastNlMeansDenoising (was 30-60s per image).

4. _scale_for_processing(): shared downscaler helper.

5. _build_foreground_mask stays in detect.py at full resolution — downscaling it
   caused mask artifacts that broke the fill check for some coins.

All other helpers unchanged from original.
"""

import cv2
import numpy as np

MAX_PROCESS_DIM = 800


def _scale_for_processing(image: np.ndarray) -> tuple:
    """
    Downscale so the largest dimension <= MAX_PROCESS_DIM.
    Returns (scaled_image, scale_factor). scale_factor==1 if already small.
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim <= MAX_PROCESS_DIM:
        return image, 1.0
    scale = MAX_PROCESS_DIM / max_dim
    small = cv2.resize(image, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA)
    return small, scale


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def reduce_noise(image: np.ndarray, method: str = "gaussian", ksize: int = 5) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1
    if method == "gaussian":
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif method == "median":
        return cv2.medianBlur(image, ksize)
    elif method == "bilateral":
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    else:
        raise ValueError(f"Unsupported noise reduction method: {method}")


def enhance_contrast(image: np.ndarray, method: str = "clahe",
                     color_space: str = "hsv") -> np.ndarray:
    if color_space == "hsv":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_clahe = clahe.apply(v)
        return cv2.cvtColor(cv2.merge((h, s, v_clahe)), cv2.COLOR_HSV2BGR)
    return image


def threshold_image(image: np.ndarray, method: str = "otsu",
                    invert: bool = False) -> np.ndarray:
    if len(image.shape) == 3:
        image = to_grayscale(image)
    if method == "otsu":
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, thresh = cv2.threshold(image, 0, 255, thresh_type + cv2.THRESH_OTSU)
        return thresh
    elif method == "adaptive":
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     thresh_type, 11, 2)
    else:
        raise ValueError(f"Unsupported threshold method: {method}")


def morphological_operations(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)


def connected_component_filter(image: np.ndarray, min_area: int = 500) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
    cleaned = np.zeros_like(image)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def edge_detection(image: np.ndarray) -> np.ndarray:
    return cv2.Canny(image, 80, 150)


def build_enhanced_edges(image: np.ndarray) -> np.ndarray:
    """
    Build a clean, gap-filled edge map for false-positive filtering.

    Used by arc_coverage and edge_density checks inside filter_false_positives.
    NOT used as HoughCircles input (HoughCircles needs raw grayscale for
    correct internal gradient direction).

    Performance: downscales to MAX_PROCESS_DIM before bilateralFilter(d=13).
    Coin boundaries are large-scale features that survive the scale change.
    Edges are upscaled back to original size with NEAREST NEIGHBOR.

    Pipeline (at reduced scale):
    1. CLAHE                — boost coin-boundary contrast.
    2. bilateralFilter(d=13) — destroy fine texture, preserve coin edges.
    3. GaussianBlur(9x9)   — remove remaining noise.
    4. Canny(15, 50)        — only large-scale edges survive.
    5. Dilate(3x3, x2)     — thicken arc by ~2 px.
    6. Close(7x7, x2)      — bridge glare/low-contrast gaps up to ~7 px.
    7. Resize back          — NEAREST NEIGHBOR preserves binary edges.
    """
    orig_h, orig_w = image.shape[:2]
    small, scale = _scale_for_processing(image)

    enhanced = enhance_contrast(small, method="clahe")
    gray = to_grayscale(enhanced)
    bilateral = cv2.bilateralFilter(gray, d=13, sigmaColor=90, sigmaSpace=90)
    over_smoothed = cv2.GaussianBlur(bilateral, (9, 9), 2.5)
    edges = cv2.Canny(over_smoothed, 15, 50)

    k_thin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edges, k_thin, iterations=2)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, k_close, iterations=2)

    if scale < 1.0:
        closed = cv2.resize(closed, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return closed


def compute_background_edge_density(image: np.ndarray, border_px: int = 80) -> float:
    """
    Measure mean edge density in the image border strip at FULL resolution.

    IMPORTANT: Must run at full resolution. Downscaling destroys fine textile
    texture (denim weave, burlap fibers) that makes the measurement meaningful.
    We process ONLY the border strips (not the whole image) for speed:
    ~4 * border_px * perimeter pixels vs. full image size.

    Uses bilateralFilter(d=9) — faster than d=13, sufficient for strips.

    Returns float in [0, 1]: fraction of border-strip pixels that are edges.

    Typical values
    --------------
    Plain paper / white:   0.00-0.03
    Concrete:              0.04-0.09
    Wood grain (test_6):   0.08-0.15
    Denim (test_10):       0.18-0.28
    Burlap (test_2):       0.12-0.20
    """
    h, w = image.shape[:2]
    bp = min(border_px, h // 3, w // 3)

    strips = []
    if bp > 0:
        strips = [
            image[:bp, :],
            image[max(0, h - bp):, :],
            image[bp:max(bp + 1, h - bp), :bp],
            image[bp:max(bp + 1, h - bp), max(0, w - bp):],
        ]
    else:
        strips = [image]

    edge_count = 0
    pixel_count = 0
    for strip in strips:
        if strip.size == 0:
            continue
        enh = enhance_contrast(strip, method="clahe")
        gray = to_grayscale(enh)
        bil = cv2.bilateralFilter(gray, d=9, sigmaColor=90, sigmaSpace=90)
        blr = cv2.GaussianBlur(bil, (7, 7), 2.0)
        edg = cv2.Canny(blr, 15, 50)
        edge_count += int(np.count_nonzero(edg))
        pixel_count += edg.size

    return edge_count / pixel_count if pixel_count > 0 else 0.0


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Return a smoothed grayscale image for cv2.HoughCircles.

    fastNlMeansDenoising removed — bilateral + Gaussian is sufficient and
    avoids the 30-60s per-image overhead.
    """
    enhanced = enhance_contrast(image, method="clahe")
    gray = to_grayscale(enhanced)
    bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    return cv2.GaussianBlur(bilateral, (7, 7), 2.0)