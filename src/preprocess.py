"""Preprocessing helpers for the coin pipeline."""

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


def enhance_contrast(image: np.ndarray, method: str = "clahe",
                     color_space: str = "hsv") -> np.ndarray:
    if color_space == "hsv":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_clahe = clahe.apply(v)
        return cv2.cvtColor(cv2.merge((h, s, v_clahe)), cv2.COLOR_HSV2BGR)
    return image


def build_enhanced_edges(image: np.ndarray) -> np.ndarray:
    """Build the edge map used by the detector's false-positive checks."""
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
    """Measure how edge-heavy the image border is at full resolution."""
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
    """Return a smoothed grayscale image for cv2.HoughCircles."""
    enhanced = enhance_contrast(image, method="clahe")
    gray = to_grayscale(enhanced)
    bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    return cv2.GaussianBlur(bilateral, (7, 7), 2.0)