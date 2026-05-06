"""
features.py — Feature Extraction
Person 3: Feature Extraction & Training

Responsibilities:
- Extract feature vectors from detected coin candidates

Features:
    Size:    radius, area, perimeter
    Shape:   circularity, aspect ratio, solidity
    Color:   HSV mean values, color histogram
    Texture: edge density, gradient variations
"""

import cv2
import numpy as np


def extract_size_features(image_roi: np.ndarray, radius: float) -> dict:
    """
    Extract size-based features.

    Returns:
        Dict with keys: radius, area, perimeter
    """
    pass


def extract_shape_features(contour) -> dict:
    """
    Extract shape-based features from a contour.

    Returns:
        Dict with keys: circularity, aspect_ratio, solidity
    """
    pass


def extract_color_features(image_roi: np.ndarray) -> dict:
    """
    Extract color-based features from a coin ROI (BGR image).

    Returns:
        Dict with keys: hsv_mean (array), color_histogram (array)
    """
    pass


def extract_texture_features(image_roi: np.ndarray) -> dict:
    """
    Extract texture-based features from a coin ROI.

    Returns:
        Dict with keys: edge_density, gradient_variation
    """
    pass


def extract_features(image: np.ndarray, circle: tuple) -> np.ndarray:
    """
    Extract a full feature vector for a single coin candidate.

    Args:
        image: Original BGR image.
        circle: (x, y, radius) tuple.

    Returns:
        1D numpy array of concatenated features.
    """
    pass
import cv2
import numpy as np


def extract_features(image, circle):
    """
    Extract a feature vector from a detected coin region.

    Parameters
    ----------
    image  : BGR image (full frame)
    circle : (x, y, radius) tuple from Hough detection

    Returns
    -------
    features : 1-D numpy array of floats, or None if region is invalid
    """
    x, y, r = int(circle[0]), int(circle[1]), int(circle[2])

    h, w = image.shape[:2]
    x1 = max(x - r, 0)
    y1 = max(y - r, 0)
    x2 = min(x + r, w)
    y2 = min(y + r, h)

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    mask = _circular_mask(roi, x - x1, y - y1, r)

    size_feats = _size_features(r, mask)
    shape_feats = _shape_features(roi, mask)
    color_feats = _color_features(roi, mask)
    texture_feats = _texture_features(roi, mask)

    return np.concatenate([size_feats, shape_feats, color_feats, texture_feats])


# Internal helpers
def _circular_mask(roi, cx, cy, r):
    """Binary mask (uint8) that is 255 inside the circle."""
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


def _size_features(radius, mask):
    """
    Size features:
      [0] radius (px)
      [1] area   (px²)
      [2] perimeter (px, approximated from mask contour)
    """
    area = float(np.pi * radius * radius)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = cv2.arcLength(contours[0], True) if contours else 2.0 * np.pi * radius

    return np.array([float(radius), area, perimeter], dtype=np.float32)


def _shape_features(roi, mask):
    """
    Shape features:
      [0] circularity  = 4π·area / perimeter²
      [1] aspect_ratio (bounding rect width / height)
      [2] solidity     = contour_area / convex_hull_area
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(3, dtype=np.float32)

    cnt = contours[0]
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    circularity = (4.0 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    _, _, bw, bh = cv2.boundingRect(cnt)
    aspect_ratio = float(bw) / bh if bh > 0 else 1.0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    return np.array([circularity, aspect_ratio, solidity], dtype=np.float32)


def _color_features(roi, mask):
    """
    Color features (HSV):
      [0-2]  mean H, S, V
      [3-18] HSV histogram (combined, 16 bins total — 6 H + 5 S + 5 V)
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mean_hsv = cv2.mean(hsv, mask=mask)[:3]

    h_hist = cv2.calcHist([hsv], [0], mask, [6], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], mask, [5], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], mask, [5], [0, 256]).flatten()

    pixel_count = float(np.count_nonzero(mask)) + 1e-6
    h_hist = h_hist / pixel_count
    s_hist = s_hist / pixel_count
    v_hist = v_hist / pixel_count

    return np.array([*mean_hsv, *h_hist, *s_hist, *v_hist], dtype=np.float32)


def _texture_features(roi, mask):
    """
    Texture features:
      [0] edge_density   = fraction of edge pixels inside the coin
      [1] gradient_mean  = mean gradient magnitude
      [2] gradient_std   = std of gradient magnitude
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    coin_pixels = float(np.count_nonzero(mask)) + 1e-6
    edge_pixels = float(np.count_nonzero(cv2.bitwise_and(edges, edges, mask=mask)))
    edge_density = edge_pixels / coin_pixels

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    coin_vals = magnitude[mask > 0]
    grad_mean = float(coin_vals.mean()) if coin_vals.size > 0 else 0.0
    grad_std = float(coin_vals.std()) if coin_vals.size > 0 else 0.0

    return np.array([edge_density, grad_mean, grad_std], dtype=np.float32)


# Batch helper (used by train.py and predict.py)

def extract_features_batch(image, circles):
    """
    Extract features for every circle in *circles* (Nx3 array).

    Returns (feature_matrix, valid_indices) so callers know which circles
    produced valid feature vectors.
    """
    feature_list = []
    valid_indices = []

    for i, circle in enumerate(circles):
        fv = extract_features(image, circle)
        if fv is not None:
            feature_list.append(fv)
            valid_indices.append(i)

    if feature_list:
        return np.vstack(feature_list), valid_indices
    return np.empty((0,), dtype=np.float32), []
