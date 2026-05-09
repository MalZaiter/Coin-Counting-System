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
    area = np.pi * radius ** 2
    perimeter = 2 * np.pi * radius
    return {
        "radius": float(radius),
        "area": float(area),
        "perimeter": float(perimeter),
    }


def extract_shape_features(contour) -> dict:
    """
    Extract shape-based features from a contour.

    Returns:
        Dict with keys: circularity, aspect_ratio, solidity
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
    else:
        circularity = 0.0

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / float(h) if h > 0 else 1.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / float(hull_area) if hull_area > 0 else 0.0

    return {
        "circularity": float(np.clip(circularity, 0.0, 1.0)),
        "aspect_ratio": float(aspect_ratio),
        "solidity": float(np.clip(solidity, 0.0, 1.0)),
    }


def extract_color_features(image_roi: np.ndarray) -> dict:
    """
    Extract color-based features from a coin ROI (BGR image).

    Returns:
        Dict with keys: hsv_mean (array), color_histogram (array)
    """
    if image_roi.ndim < 3 or image_roi.size == 0:
        return {
            "hsv_mean": np.zeros(3, dtype=np.float32),
            "color_histogram": np.zeros(48, dtype=np.float32),
        }

    hsv = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)

    # Build a circular mask so we only sample the coin, not the background
    h, w = image_roi.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = min(cx, cy)
    cv2.circle(mask, (cx, cy), max(r - 2, 1), 255, -1)

    hsv_mean = cv2.mean(hsv, mask=mask)[:3]

    # 16-bin hue histogram + 16-bin saturation histogram + 16-bin value histogram
    h_hist = cv2.calcHist([hsv], [0], mask, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], mask, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], mask, [16], [0, 256]).flatten()

    # Normalize each histogram independently
    def _norm(hist):
        s = hist.sum()
        return hist / s if s > 0 else hist

    color_histogram = np.concatenate([_norm(h_hist), _norm(s_hist), _norm(v_hist)])

    return {
        "hsv_mean": np.array(hsv_mean, dtype=np.float32),
        "color_histogram": color_histogram.astype(np.float32),
    }


def extract_texture_features(image_roi: np.ndarray) -> dict:
    """
    Extract texture-based features from a coin ROI.

    Returns:
        Dict with keys: edge_density, gradient_variation
    """
    if image_roi.size == 0:
        return {"edge_density": 0.0, "gradient_variation": 0.0}

    if image_roi.ndim == 3:
        gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_roi.copy()

    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = min(cx, cy)
    cv2.circle(mask, (cx, cy), max(r - 2, 1), 255, -1)

    # Edge density: fraction of Canny edge pixels inside the coin disk
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, 30, 90)
    coin_pixels = int(np.count_nonzero(mask))
    edge_pixels = int(np.count_nonzero(cv2.bitwise_and(edges, edges, mask=mask)))
    edge_density = edge_pixels / coin_pixels if coin_pixels > 0 else 0.0

    # Gradient variation: std dev of gradient magnitude inside the coin disk
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    coin_grad = grad_mag[mask > 0]
    gradient_variation = float(np.std(coin_grad)) if coin_grad.size > 0 else 0.0

    return {
        "edge_density": float(edge_density),
        "gradient_variation": float(gradient_variation),
    }


def extract_features(image: np.ndarray, circle: tuple) -> np.ndarray:
    """
    Extract a full feature vector for a single coin candidate.

    Args:
        image:  Original BGR image.
        circle: (x, y, radius) tuple.

    Returns:
        1D numpy array of concatenated features.
    """
    from src.utils import crop_roi

    x, y, radius = int(circle[0]), int(circle[1]), int(circle[2])

    roi = crop_roi(image, x, y, radius)

    # ── Size features ────────────────────────────────────────────────────────
    size_feats = extract_size_features(roi, radius)
    size_vec = np.array([
        size_feats["radius"],
        size_feats["area"],
        size_feats["perimeter"],
    ], dtype=np.float32)

    # ── Shape features — approximate the coin boundary as a polygon contour ─
    h_img, w_img = image.shape[:2]
    n_pts = 72
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    pts = np.array([
        [
            int(np.clip(x + radius * np.cos(a), 0, w_img - 1)),
            int(np.clip(y + radius * np.sin(a), 0, h_img - 1)),
        ]
        for a in angles
    ], dtype=np.int32).reshape((-1, 1, 2))
    shape_feats = extract_shape_features(pts)
    shape_vec = np.array([
        shape_feats["circularity"],
        shape_feats["aspect_ratio"],
        shape_feats["solidity"],
    ], dtype=np.float32)

    # ── Color features ───────────────────────────────────────────────────────
    color_feats = extract_color_features(roi)
    color_vec = np.concatenate([
        color_feats["hsv_mean"],       # 3 values
        color_feats["color_histogram"] # 48 values
    ])

    # ── Texture features ─────────────────────────────────────────────────────
    texture_feats = extract_texture_features(roi)
    texture_vec = np.array([
        texture_feats["edge_density"],
        texture_feats["gradient_variation"],
    ], dtype=np.float32)

    feature_vector = np.concatenate([size_vec, shape_vec, color_vec, texture_vec])
    return feature_vector.astype(np.float32)