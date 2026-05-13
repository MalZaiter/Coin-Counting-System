"""
utils.py — Helper / Utility Functions

Shared utilities used across the pipeline.
"""

import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk as a BGR numpy array.

    Raises:
        FileNotFoundError if the path does not exist.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image


def crop_roi(image: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    """
    Crop a square region of interest around a detected circle.

    Returns:
        Cropped image patch.
    """
    h, w = image.shape[:2]
    x1 = max(x - radius, 0)
    y1 = max(y - radius, 0)
    x2 = min(x + radius, w)
    y2 = min(y + radius, h)

    roi = image[y1:y2, x1:x2]
    return roi


def normalize_features(features: np.ndarray, scaler=None):
    """
    Normalize a feature matrix.

    If scaler is None, fit a new StandardScaler and return (scaled_X, scaler).
    If scaler is provided, transform using the existing scaler.

    Returns:
        (scaled_features, scaler)
    """
    features = np.asarray(features)
    if scaler is None:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
    else:
        scaled_features = scaler.transform(features)

    return scaled_features, scaler
