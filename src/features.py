"""Feature extraction for coin classification."""

import cv2
import numpy as np


def _empty_color_features() -> dict:
    return {
        "hsv_mean": np.zeros(3, dtype=np.float32),
        "color_histogram": np.zeros(48, dtype=np.float32),
    }


def _empty_texture_features() -> dict:
    return {"edge_density": 0.0, "gradient_variation": 0.0}


def _ensure_bgr(image_roi: np.ndarray):
    if image_roi.size == 0:
        return None
    if image_roi.ndim == 2:
        return cv2.cvtColor(image_roi, cv2.COLOR_GRAY2BGR)
    if image_roi.ndim == 3 and image_roi.shape[2] == 1:
        return cv2.cvtColor(image_roi[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image_roi.ndim == 3 and image_roi.shape[2] == 3:
        return image_roi
    return None


def _coin_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = max(min(center) - 2, 1)
    cv2.circle(mask, center, radius, 255, -1)
    return mask


def _masked_gray(image_roi: np.ndarray) -> tuple:
    image_roi = _ensure_bgr(image_roi)
    if image_roi is None:
        return None, None

    gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    mask = _coin_mask(*gray.shape[:2])
    return gray, mask


def _safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size > 0 else 0.0


def _spatial_grid_features(gray: np.ndarray, mask: np.ndarray, grid_size: int = 4) -> np.ndarray:
    h, w = gray.shape[:2]
    cell_h = max(h // grid_size, 1)
    cell_w = max(w // grid_size, 1)

    features = []
    for row in range(grid_size):
        y1 = row * cell_h
        y2 = h if row == grid_size - 1 else (row + 1) * cell_h
        for col in range(grid_size):
            x1 = col * cell_w
            x2 = w if col == grid_size - 1 else (col + 1) * cell_w

            cell_mask = mask[y1:y2, x1:x2] > 0
            if not np.any(cell_mask):
                features.extend([0.0, 0.0])
                continue

            cell_gray = gray[y1:y2, x1:x2][cell_mask]
            cell_edges = cv2.Canny(gray[y1:y2, x1:x2], 30, 90)
            cell_edge_count = np.count_nonzero(cell_edges[cell_mask])
            cell_area = max(int(np.count_nonzero(cell_mask)), 1)

            features.append(float(cell_gray.mean()) / 255.0)
            features.append(float(cell_edge_count) / float(cell_area))

    return np.asarray(features, dtype=np.float32)


def _radial_profile_features(gray: np.ndarray, mask: np.ndarray, bands: int = 4) -> np.ndarray:
    h, w = gray.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    max_radius = min(cx, cy) - 1.0
    if max_radius <= 0:
        return np.zeros(bands * 2, dtype=np.float32)

    yy, xx = np.indices((h, w))
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    normalized = np.clip(dist / max_radius, 0.0, 0.9999)

    features = []
    for band in range(bands):
        band_mask = (normalized >= band / bands) & (normalized < (band + 1) / bands) & (mask > 0)
        if not np.any(band_mask):
            features.extend([0.0, 0.0])
            continue

        band_gray = gray[band_mask]
        band_edges = cv2.Canny(gray, 30, 90)[band_mask]
        features.append(float(band_gray.mean()) / 255.0)
        features.append(float(np.count_nonzero(band_edges)) / float(max(int(np.count_nonzero(band_mask)), 1)))

    return np.asarray(features, dtype=np.float32)


def extract_color_features(image_roi: np.ndarray) -> dict:
    image_roi = _ensure_bgr(image_roi)
    if image_roi is None:
        return _empty_color_features()

    hsv = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)
    mask = _coin_mask(*image_roi.shape[:2])

    def _normalize(hist: np.ndarray) -> np.ndarray:
        total = hist.sum()
        return hist / total if total > 0 else hist

    hsv_mean = np.array(cv2.mean(hsv, mask=mask)[:3], dtype=np.float32)
    color_histogram = np.concatenate([
        _normalize(cv2.calcHist([hsv], [0], mask, [16], [0, 180]).flatten()),
        _normalize(cv2.calcHist([hsv], [1], mask, [16], [0, 256]).flatten()),
        _normalize(cv2.calcHist([hsv], [2], mask, [16], [0, 256]).flatten()),
    ]).astype(np.float32)

    return {"hsv_mean": hsv_mean, "color_histogram": color_histogram}


def extract_texture_features(image_roi: np.ndarray) -> dict:
    gray, mask = _masked_gray(image_roi)
    if gray is None or mask is None:
        return _empty_texture_features()

    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, 30, 90)
    coin_pixels = max(int(np.count_nonzero(mask)), 1)
    edge_pixels = int(np.count_nonzero(cv2.bitwise_and(edges, edges, mask=mask)))
    edge_density = edge_pixels / coin_pixels

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_variation = float(np.std(np.sqrt(grad_x ** 2 + grad_y ** 2)[mask > 0]))

    return {"edge_density": float(edge_density), "gradient_variation": gradient_variation}


def extract_spatial_features(image_roi: np.ndarray) -> np.ndarray:
    """Extract compact spatial layout features from the coin interior."""
    gray, mask = _masked_gray(image_roi)
    if gray is None or mask is None:
        return np.zeros(40, dtype=np.float32)

    grid_vec = _spatial_grid_features(gray, mask, grid_size=4)
    radial_vec = _radial_profile_features(gray, mask, bands=4)
    return np.concatenate([grid_vec, radial_vec]).astype(np.float32)


def extract_features(image: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    Extract a full feature vector for a single coin candidate.

    Args:
        image: Original BGR image.
        bbox:  Either (x, y, radius) for circle OR (x1, y1, x2, y2) for bounding box.
               Auto-detects format based on length.

    Returns:
        1D numpy array of concatenated features.
    """
    # Auto-detect format: circle has 3 elements, bbox has 4
    if len(bbox) == 3:
        # Circle format: (x, y, radius)
        x, y, radius = int(bbox[0]), int(bbox[1]), int(bbox[2])
        from src.utils import crop_roi
        roi = crop_roi(image, x, y, radius)
        size_vec = np.array([radius, np.pi * radius ** 2, 2 * np.pi * radius], dtype=np.float32)
    else:
        # Bounding box format: (x1, y1, x2, y2)
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            roi = image[0:1, 0:1]
        
        w = x2 - x1
        h = y2 - y1
        size_vec = np.array([max(w, h) / 2, w * h, 2 * (w + h)], dtype=np.float32)
        x, y = (x1 + x2) // 2, (y1 + y2) // 2
        radius = max(w, h) // 2

    # Approximate circle for shape features
    circle_contour = np.array([
        [
            int(np.clip(x + radius * np.cos(angle), 0, image.shape[1] - 1)),
            int(np.clip(y + radius * np.sin(angle), 0, image.shape[0] - 1)),
        ]
        for angle in np.linspace(0, 2 * np.pi, 72, endpoint=False)
    ], dtype=np.int32).reshape((-1, 1, 2))
    contour_area = cv2.contourArea(circle_contour)
    contour_perimeter = cv2.arcLength(circle_contour, True)
    circularity = 4 * np.pi * contour_area / (contour_perimeter ** 2) if contour_perimeter > 0 else 0.0
    x0, y0, w0, h0 = cv2.boundingRect(circle_contour)
    hull_area = cv2.contourArea(cv2.convexHull(circle_contour))
    shape_vec = np.array([
        float(np.clip(circularity, 0.0, 1.0)),
        float(w0) / float(h0) if h0 > 0 else 1.0,
        float(contour_area) / float(hull_area) if hull_area > 0 else 0.0,
    ], dtype=np.float32)

    color_feats = extract_color_features(roi)
    color_vec = np.concatenate([color_feats["hsv_mean"], color_feats["color_histogram"]]).astype(np.float32)

    texture_feats = extract_texture_features(roi)
    texture_vec = np.array([texture_feats["edge_density"], texture_feats["gradient_variation"]], dtype=np.float32)
    spatial_vec = extract_spatial_features(roi)

    feature_vector = np.concatenate([size_vec, shape_vec, color_vec, texture_vec, spatial_vec])
    return feature_vector.astype(np.float32)