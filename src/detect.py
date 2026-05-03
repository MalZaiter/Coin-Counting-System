"""
detect.py — Circle Detection & Filtering
Person 2: Detection & Filtering

Responsibilities:
- Detect circular objects using Hough Transform
- Remove false positives (e.g., bottle caps)

Techniques:
- cv2.HoughCircles
- Contour analysis: circularity, solidity, edge smoothness
- Edge detection (Canny)
"""

import cv2
import numpy as np


def detect_circles(image: np.ndarray) -> list:
    """
    Detect circular objects in a preprocessed image using HoughCircles.

    Returns:
        List of (x, y, radius) tuples for each detected circle.
    """
    pass


def compute_circularity(contour) -> float:
    """
    Compute the circularity of a contour.
    Circularity = 4 * pi * area / perimeter^2
    Perfect circle = 1.0
    """
    pass


def compute_solidity(contour) -> float:
    """
    Compute the solidity of a contour.
    Solidity = contour area / convex hull area
    """
    pass


def filter_false_positives(image: np.ndarray, circles: list) -> list:
    """
    Filter out non-coin circular objects (e.g., bottle caps) using:
    - Circularity threshold
    - Solidity threshold
    - Edge smoothness analysis

    Args:
        image: Original or preprocessed image.
        circles: List of (x, y, radius) from detect_circles.

    Returns:
        Filtered list of valid coin candidates.
    """
    pass


def detect_coins(image: np.ndarray) -> list:
    """
    Full detection pipeline:
    1. Detect circles via HoughCircles
    2. Filter false positives via contour analysis

    Returns:
        List of valid coin candidates as (x, y, radius) tuples.
    """
    pass
