"""
utils.py — Helper / Utility Functions

Shared utilities used across the pipeline.
"""

import os
import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk as a BGR numpy array.

    Raises:
        FileNotFoundError if the path does not exist.
    """
    pass


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save a numpy image array to disk.
    """
    pass


def show_image(window_name: str, image: np.ndarray) -> None:
    """
    Display an image in an OpenCV window (blocking until key press).
    """
    pass


def crop_roi(image: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    """
    Crop a square region of interest around a detected circle.

    Returns:
        Cropped image patch.
    """
    pass


def list_images(directory: str, extensions: tuple = (".jpg", ".jpeg", ".png")) -> list:
    """
    Return a sorted list of image file paths within a directory.
    """
    pass


def normalize_features(features: np.ndarray, scaler=None):
    """
    Normalize a feature matrix.

    If scaler is None, fit a new StandardScaler and return (scaled_X, scaler).
    If scaler is provided, transform using the existing scaler.

    Returns:
        (scaled_features, scaler)
    """
    pass
