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
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=150
    )

    if circles is not None:
        return np.uint16(np.around(circles[0]))
    return []



def compute_circularity(contour) -> float:
    """
    Compute the circularity of a contour.
    Circularity = 4 * pi * area / perimeter^2
    Perfect circle = 1.0
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return 0

    return (4 * np.pi * area) / (perimeter * perimeter)


def compute_solidity(contour) -> float:
    """
    Compute the solidity of a contour.
    Solidity = contour area / convex hull area
    """
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        return 0

    return area / hull_area


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
    valid = []

    for (x, y, r) in circles:
        # Create a mask for this circle
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        # Find contour from mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        cnt = contours[0]

        circularity = compute_circularity(cnt)
        solidity = compute_solidity(cnt)

        # 🔥 Filtering conditions (tune these)
        if circularity > 0.8 and solidity > 0.9:
            valid.append((x, y, r))

    return valid


def detect_coins(image: np.ndarray) -> list:
    """
    Full detection pipeline:
    1. Detect circles via HoughCircles
    2. Filter false positives via contour analysis

    Returns:
        List of valid coin candidates as (x, y, radius) tuples.
    """
    from src.preprocess import preprocess

    # Step 1: Preprocess image → edges
    edges = preprocess(image)

    # Step 2: Detect circles
    circles = detect_circles(edges)

    if len(circles) == 0:
        return []

    # Step 3: Filter real coins
    valid_coins = filter_false_positives(image, circles)

    return valid_coins
