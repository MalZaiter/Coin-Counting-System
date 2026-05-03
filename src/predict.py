"""
predict.py — Inference on New Images
Person 4: Integration & Evaluation

Responsibilities:
- Load trained model
- Run full pipeline on new images
- Draw detected coins and labels on output image
- Count total coins and compute total value
"""

import cv2
import numpy as np


def load_model(model_path: str, scaler_path: str = None):
    """
    Load the trained classifier (and optional scaler) from disk.

    Returns:
        (model, scaler) tuple. scaler may be None.
    """
    pass


def predict_coin(feature_vector: np.ndarray, model, scaler=None) -> str:
    """
    Predict the coin type for a single feature vector.

    Returns:
        Predicted label string (e.g., 'quarter', 'dime').
    """
    pass


def draw_results(image: np.ndarray, coins: list, labels: list) -> np.ndarray:
    """
    Draw detected coin circles and their predicted labels on the image.

    Args:
        image:  Original BGR image.
        coins:  List of (x, y, radius) tuples.
        labels: Corresponding predicted label for each coin.

    Returns:
        Annotated image.
    """
    pass


def count_coins(labels: list) -> dict:
    """
    Count occurrences of each coin type.

    Returns:
        Dict mapping coin label -> count (e.g., {'quarter': 3, 'dime': 2}).
    """
    pass


def compute_total_value(label_counts: dict) -> float:
    """
    Compute total monetary value from coin counts.

    Returns:
        Total value in dollars (float).
    """
    pass


def predict(image_path: str, model_path: str, scaler_path: str = None) -> dict:
    """
    Full prediction pipeline for a single image:
    1. Load and preprocess image
    2. Detect coins
    3. Extract features
    4. Predict coin types
    5. Draw and return results

    Returns:
        Dict with keys: annotated_image, label_counts, total_value
    """
    pass
