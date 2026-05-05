"""
preprocess.py — Image Preprocessing
Person 1: Data & Preprocessing

Responsibilities:
- Grayscale conversion
- Noise reduction (Gaussian / Median blur)
- Histogram equalization/normalization
- Edge detection (Canny)
"""

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""
    if len(image.shape) == 2:
        return image #already greyscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pass


def reduce_noise(image: np.ndarray, method: str = "gaussian") -> np.ndarray:
    """
    Reduce noise using Gaussian or Median blur.

    Args:
        image: Grayscale or BGR image.
        method: 'gaussian' or 'median'.
    """
    if method == "gaussian":
        return cv2.GaussianBlur(image, (11,11), 0)
    elif method == "median":
        return cv2.medianBlur(image, 11)
    else:
        raise ValueError(f"Unsupported noise reduction method: {method}")
    pass


def equalize_normalize(image: np.ndarray, method: str = "equalize") -> np.ndarray:
    """
    Apply Histogram Equalization or normalization to enhance contrast.
    """
    if len(image.shape) == 3:
        image = to_grayscale(image)

    if method == "equalize":
        return cv2.equalizeHist(image)
    elif method == "normalize":
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    else:
        raise ValueError(f"Unsupported histogram equalization method: {method}")
    
# def apply_CLAHE(image: np.ndarray) -> np.ndarray:
#     """
#     Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast.
#     """
#     if len(image.shape) == 3:
#         image = to_grayscale(image)

#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     return clahe.apply(image)


# def threshold_image(image: np.ndarray, method: str = "adaptive") -> np.ndarray:
#     """
#     Threshold a grayscale image.

#     Args:
#         image: Grayscale image.
#         method: 'global' or 'adaptive'.
#     """
#     if method == "global":
#         _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
#         return thresh
#     elif method == "adaptive":
#         return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
#     else:
#         raise ValueError(f"Method must be global or adaptive")


# def morphological_operations(image: np.ndarray) -> np.ndarray:
#     """
#     Apply morphological operations (e.g., opening/closing) to clean up
#     binary mask.
#     """
#     kernel = np.ones((9,9), np.uint8)
#     #remove noise (opening)
#     opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
#     #close holes inside objects (closing)
#     closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
#     return closing

def edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Apply Canny edge detection to find edges in the image.
    """
    return cv2.Canny(image, 80, 140)


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Grayscale
    2. Noise reduction
    3. Histogram equalization/normalization
    4. Edge detection

    Returns preprocessed image ready for circle detection.
    """
    gray = to_grayscale(image)
    denoised = reduce_noise(gray, method="median")
    equalized = equalize_normalize(denoised, method="normalize")
    # thresh = threshold_image(equalized, method="global")
    # cleaned = morphological_operations(thresh)
    edges = edge_detection(equalized)

    return edges
