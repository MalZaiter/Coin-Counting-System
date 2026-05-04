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


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save a numpy image array to disk.
    """
    #create dictionary
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    success = cv2.imwrite(output_path, image)
    if not success:
        raise IOError(f"Failed to save image: {output_path}")
    pass


def show_image(window_name: str, image: np.ndarray) -> None:
    """
    Display an image in an OpenCV window (blocking until key press).
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


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


def list_images(directory: str, extensions: tuple = (".jpg", ".jpeg", ".png")) -> list:
    """
    Return a sorted list of image file paths within a directory.
    """
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
    return sorted(image_paths)


def normalize_features(features: np.ndarray, scaler=None):
    """
    Normalize a feature matrix.

    If scaler is None, fit a new StandardScaler and return (scaled_X, scaler).
    If scaler is provided, transform using the existing scaler.

    Returns:
        (scaled_features, scaler)
    """
    features = np.array(features)
    if scaler is None:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
    else:
        scaled_features = scaler.transform(features)
    
    return scaled_features, scaler
