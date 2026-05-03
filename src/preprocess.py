"""
preprocess.py — Image Preprocessing
Person 1: Data & Preprocessing

Responsibilities:
- Grayscale conversion
- Noise reduction (Gaussian / Median blur)
- Histogram equalization (CLAHE)
- Thresholding (global + adaptive)
- Morphological operations
"""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""
    pass


def reduce_noise(image: np.ndarray, method: str = "gaussian") -> np.ndarray:
    """
    Reduce noise using Gaussian or Median blur.

    Args:
        image: Grayscale or BGR image.
        method: 'gaussian' or 'median'.
    """
    pass


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to improve contrast under varying lighting conditions.
    """
    pass


def threshold_image(image: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Threshold a grayscale image.

    Args:
        image: Grayscale image.
        method: 'global' or 'adaptive'.
    """
    pass


def morphological_operations(image: np.ndarray) -> np.ndarray:
    """
    Apply morphological operations (e.g., opening/closing) to clean up
    binary mask.
    """
    pass


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Grayscale
    2. Noise reduction
    3. CLAHE
    4. Thresholding
    5. Morphological cleanup

    Returns preprocessed image ready for circle detection.
    """
    pass
