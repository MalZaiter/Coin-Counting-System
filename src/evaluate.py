"""
evaluate.py — Evaluation Metrics
Person 4: Integration & Evaluation

Responsibilities:
- Measure classification accuracy
- Generate confusion matrix
- Report false positive / false negative rates
"""

import numpy as np
import os
from pathlib import Path
from src.predict import load_model


def compute_accuracy(y_true: list, y_pred: list) -> float:
    """
    Compute overall classification accuracy.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_pred)


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
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred, labels=labels)


def print_classification_report(y_true: list, y_pred: list, labels: list) -> None:
    """
    Print precision, recall, and F1-score for each class.
    """
    from sklearn.metrics import classification_report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))


def plot_confusion_matrix(matrix: np.ndarray, labels: list, save_path: str = None) -> None:
    """
    Visualize the confusion matrix using matplotlib.

    Args:
        matrix: 2D confusion matrix array
        labels: List of class labels
        save_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        plt.close()
    except Exception as e:
        print(f"Could not plot confusion matrix: {e}")


def evaluate(test_images_dir: str, labels_path: str, model_path: str, scaler_path: str = None, plot: bool = True) -> dict:
    """
    Full evaluation pipeline:
    1. Load test images and ground truth labels
    2. Run predictions
    3. Compute and display metrics

    Args:
        test_images_dir: Directory with test images
        labels_path: Path to labels CSV file
        model_path: Path to trained model
        scaler_path: Optional path to scaler
        plot: Whether to display confusion matrix plot (default: True)

    Returns:
        Dict with keys: accuracy, confusion_matrix, report, y_true, y_pred
    """
    from src.predict import predict

    # Load model
    model, scaler = load_model(model_path, scaler_path)

    # Load ground truth labels
    y_true = []
    image_files = []
    try:
        import csv
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_files.append(row.get('image_path'))
                y_true.append(row.get('coin_type'))
    except Exception:
        # Fallback: assume labels are in text files alongside images
        image_dir = Path(test_images_dir)
        for img_path in sorted(image_dir.glob('*.jpg')) + sorted(image_dir.glob('*.png')):
            image_files.append(img_path.name)
            label_path = img_path.with_suffix('.txt')
            if label_path.exists():
                with open(label_path, 'r') as f:
                    y_true.append(f.read().strip())
            else:
                y_true.append('unknown')

    if not image_files or not y_true:
        raise ValueError(f"No images or labels found in {test_images_dir}")
    # Remove duplicate image paths
    unique_pairs = {}

    for img, label in zip(image_files, y_true):
        if img not in unique_pairs:
            unique_pairs[img] = label

    image_files = list(unique_pairs.keys())
    y_true = list(unique_pairs.values())
#run predictions
    y_pred = []

    for i, img_file in enumerate(image_files):

        try:
            # Fix dataset path naming
            img_path = img_file.replace("archive\\", "data\\training\\")
            img_path = img_path.replace("archive/", "data/training/")
            print(f"Evaluating {i + 1}/{len(image_files)} : {img_file}")

            # Run prediction
            result = predict(
                img_path,
                model_path,
                scaler_path
            )

            # Extract predicted labels
            labels = result.get("labels", [])

            # Choose most common predicted label
            if labels:
                from collections import Counter

                most_common_label = Counter(labels).most_common(1)[0][0]
                y_pred.append(most_common_label)

            else:
                y_pred.append("unknown")

        except Exception as e:
            print(f"Error predicting {img_file}: {e}")
            y_pred.append("unknown")
    # Ensure same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    # Compute metrics
    unique_labels = sorted(set(y_true + y_pred))
    accuracy = compute_accuracy(y_true, y_pred)
    conf_matrix = compute_confusion_matrix(y_true, y_pred, unique_labels)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {len(y_true)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*60}\n")

    if plot:
        print_classification_report(y_true, y_pred, unique_labels)
        plot_confusion_matrix(conf_matrix, unique_labels)

    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'labels': unique_labels,
        'y_true': y_true,
        'y_pred': y_pred
    }


