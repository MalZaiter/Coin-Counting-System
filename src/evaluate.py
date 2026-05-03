"""
evaluate.py — Evaluation Metrics
Person 4: Integration & Evaluation

Responsibilities:
- Measure classification accuracy
- Generate confusion matrix
- Report false positive / false negative rates
"""

import numpy as np


def compute_accuracy(y_true: list, y_pred: list) -> float:
    """
    Compute overall classification accuracy.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    pass


def compute_confusion_matrix(y_true: list, y_pred: list, labels: list) -> np.ndarray:
    """
    Compute the confusion matrix.

    Args:
        y_true:  Ground truth labels.
        y_pred:  Predicted labels.
        labels:  Ordered list of class labels.

    Returns:
        2D numpy array (confusion matrix).
    """
    pass


def print_classification_report(y_true: list, y_pred: list, labels: list) -> None:
    """
    Print precision, recall, and F1-score for each class.
    """
    pass


def plot_confusion_matrix(matrix: np.ndarray, labels: list) -> None:
    """
    Visualize the confusion matrix using matplotlib.
    """
    pass


def evaluate(test_images_dir: str, labels_path: str, model_path: str) -> dict:
    """
    Full evaluation pipeline:
    1. Load test images and ground truth labels
    2. Run predictions
    3. Compute and display metrics

    Returns:
        Dict with keys: accuracy, confusion_matrix, report
    """
    pass
