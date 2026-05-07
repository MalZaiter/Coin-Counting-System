"""
preprocess.py — Image Preprocessing Utilities

Provides:
  - preprocess()            → smoothed grayscale for cv2.HoughCircles
  - build_enhanced_edges()  → gap-filled edge map for false-positive filtering
  - Supporting helpers: to_grayscale, reduce_noise, enhance_contrast,
    threshold_image, morphological_operations, connected_component_filter,
    edge_detection
"""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def reduce_noise(image: np.ndarray, method: str = "gaussian", ksize: int = 5) -> np.ndarray:
    """
    Noise reduction with Gaussian, Median, or Bilateral blur.
    Bilateral is preferred — it smooths texture while preserving coin edges.
    """
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


def enhance_contrast(image: np.ndarray, method: str = "clahe", color_space: str = "hsv") -> np.ndarray:
    """
    Contrast enhancement via CLAHE on the V channel in HSV space.
    Boosting coin-boundary contrast before edge/gradient operations.
    """
    if color_space == "hsv":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_clahe = clahe.apply(v)
        return cv2.cvtColor(cv2.merge((h, s, v_clahe)), cv2.COLOR_HSV2BGR)
    return image


def threshold_image(image: np.ndarray, method: str = "otsu", invert: bool = False) -> np.ndarray:
    """Thresholding to separate coin regions from background."""
    if len(image.shape) == 3:
        image = to_grayscale(image)

    if method == "otsu":
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, thresh = cv2.threshold(image, 0, 255, thresh_type + cv2.THRESH_OTSU)
        return thresh
    elif method == "adaptive":
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, thresh_type, 11, 2
        )
    else:
        raise ValueError(f"Unsupported threshold method: {method}")


def morphological_operations(image: np.ndarray) -> np.ndarray:
    """Morphological opening + closing to clean binary masks."""
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)


def connected_component_filter(image: np.ndarray, min_area: int = 500) -> np.ndarray:
    """Remove small noise blobs via connected component analysis."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
    cleaned = np.zeros_like(image)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def edge_detection(image: np.ndarray) -> np.ndarray:
    """Canny edge detection on a grayscale image."""
    return cv2.Canny(image, 80, 150)


def build_enhanced_edges(image: np.ndarray) -> np.ndarray:
    """
    Build a clean, gap-filled edge map for post-filter validation.

    Used by arc_coverage and edge_density checks inside filter_false_positives.
    NOT used as HoughCircles input (HoughCircles needs a raw grayscale for
    correct internal gradient direction).

    Pipeline
    --------
    1. CLAHE — boosts coin-boundary contrast.
    2. Heavy bilateral + wide Gaussian — destroys fine texture (fabric, paper,
       wood) while leaving large-scale coin-boundary gradients intact.
    3. Canny at low thresholds — only large-scale edges survive the heavy blur.
    4. Dilation (3×3, ×2) — thickens arc by ~2 px.
    5. Closing (7×7, ×2) — bridges glare/low-contrast gaps up to ~7 px wide.
    """
    enhanced = enhance_contrast(image, method="clahe")
    gray = to_grayscale(enhanced)

    bilateral = cv2.bilateralFilter(gray, d=13, sigmaColor=90, sigmaSpace=90)
    over_smoothed = cv2.GaussianBlur(bilateral, (9, 9), 2.5)
    edges = cv2.Canny(over_smoothed, 15, 50)

    k_thin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edges, k_thin, iterations=2)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, k_close, iterations=2)


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Return a smoothed grayscale image for cv2.HoughCircles.

    Why grayscale (not an edge map)
    --------------------------------
    HoughCircles (HOUGH_GRADIENT) computes the image gradient internally and
    casts votes inward along each gradient direction.  A properly smoothed
    grayscale preserves reliable coin-boundary gradient directions; a binary
    edge map destroys them (the step edge at the white band boundary is nearly
    random in direction), causing votes to scatter and accumulator peaks to
    never form.

    Texture elimination strategy
    ----------------------------
    Bilateral filter (d=9, σ=75): edge-preserving smoothing that kills
    fine-grained texture while keeping large coin-boundary gradients.
    Gaussian (7×7, σ=2.0): removes any remaining high-frequency noise.
    """
    enhanced = enhance_contrast(image, method="clahe")
    gray = to_grayscale(enhanced)
    bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    # h=10 (was 15): lighter denoising preserves weak coin-edge gradients on
    # low-contrast backgrounds (silver coins on dark fabric, concrete).
    denoised = cv2.fastNlMeansDenoising(bilateral, h=10, templateWindowSize=7, searchWindowSize=21)
    return cv2.GaussianBlur(denoised, (7, 7), 2.0)
