"""
train.py — Model Training
Person 3: Feature Extraction & Training

Responsibilities:
- Load labeled training data
- Train KNN or SVM classifier
- Save trained model and optional scaler
"""

import os
import numpy as np


def load_training_data(labels_path: str, images_dir: str) -> tuple:
    """
    Load images and labels for training.

    Args:
        labels_path: Path to labels CSV/JSON file.
        images_dir: Directory of training images.

    Returns:
        (X, y) — feature matrix and label vector.
    """
    pass


def train_knn(X: np.ndarray, y: np.ndarray, n_neighbors: int = 5):
    """
    Train a K-Nearest Neighbors classifier.

    Returns:
        Trained KNN model.
    """
    pass


def train_svm(X: np.ndarray, y: np.ndarray):
    """
    Train a Support Vector Machine classifier.

    Returns:
        Trained SVM model.
    """
    pass


def save_model(model, scaler, model_path: str, scaler_path: str) -> None:
    """
    Save the trained model and scaler to disk using pickle/joblib.
    """
    pass


def train(labels_path: str, images_dir: str, model_type: str = "svm") -> None:
    """
    Full training pipeline:
    1. Load data
    2. Extract features
    3. Scale features
    4. Train classifier
    5. Save model

    Args:
        labels_path: Path to labels file.
        images_dir:  Path to training images directory.
        model_type:  'knn' or 'svm'.
    """
    pass
