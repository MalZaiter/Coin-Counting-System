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
