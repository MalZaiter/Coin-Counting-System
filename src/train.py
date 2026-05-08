"""
train.py — Model Training
Person 3: Feature Extraction & Training

Responsibilities:
- Load labeled training data
- Train KNN or SVM classifier
- Save trained model and optional scaler
"""

import csv
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.detect import detect_coins
from src.features import extract_features
from src.utils import load_image, normalize_features


def _resolve_image_path(images_dir: str, image_ref: str) -> str:
    """Resolve an image path from a manifest entry."""
    if not image_ref:
        raise ValueError("Manifest row is missing an image path")

    candidate = Path(image_ref)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    relative_to_cwd = Path.cwd() / candidate
    if relative_to_cwd.exists():
        return str(relative_to_cwd)

    relative_to_images = Path(images_dir) / candidate
    if relative_to_images.exists():
        return str(relative_to_images)

    raise FileNotFoundError(f"Unable to resolve image path: {image_ref}")


def _load_manifest(labels_path: str) -> list:
    """Load circle-level or image-level labels from CSV or JSON."""
    manifest_path = Path(labels_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    suffix = manifest_path.suffix.lower()
    if suffix == ".json":
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            if "items" in payload and isinstance(payload["items"], list):
                return payload["items"]
            raise ValueError("JSON labels file must contain a list or an 'items' array")
        if not isinstance(payload, list):
            raise ValueError("JSON labels file must contain a list of label entries")
        return payload

    if suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader)

    raise ValueError("labels_path must point to a CSV or JSON manifest")


def _row_label(row: dict) -> str:
    """Read the target label from a manifest row."""
    for key in ("label", "coin_type", "class", "target"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    raise ValueError("Manifest row is missing a label column")


def _row_circle(row: dict):
    """Return an optional (x, y, radius) tuple from a manifest row."""
    required = ("x", "y", "radius")
    if not all(row.get(key) not in (None, "") for key in required):
        return None
    return (
        int(float(row["x"])),
        int(float(row["y"])),
        int(float(row["radius"])),
    )


def _load_directory_dataset(images_dir: str) -> list:
    """Fallback loader for class-organized folders.

    Each immediate subdirectory name is treated as the label.
    """
    dataset = []
    root = Path(images_dir)
    if not root.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        label = child.name
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            for image_path in sorted(child.rglob(pattern)):
                dataset.append({"image_path": str(image_path), "label": label})
    return dataset


def load_training_data(labels_path: str, images_dir: str) -> tuple:
    """
    Load images and labels for training.

    Args:
        labels_path: Path to labels CSV/JSON file. Use circle-level labels so
            raw coin images and complex coin/bottle-cap images can share the
            same extractor while keeping different target labels.
        images_dir: Directory used to resolve relative image paths.

    Returns:
        (X, y) — feature matrix and label vector.
    """
    if labels_path and os.path.exists(labels_path):
        entries = _load_manifest(labels_path)
    else:
        entries = _load_directory_dataset(images_dir)

    feature_rows = []
    labels = []

    for row in entries:
        image_ref = row.get("image_path") or row.get("image") or row.get("filename")
        if not image_ref:
            continue

        label = _row_label(row)
        image_path = _resolve_image_path(images_dir, image_ref)
        image = load_image(image_path)

        circle = _row_circle(row)
        circles = [circle] if circle is not None else detect_coins(image)

        for detected_circle in circles:
            feature_rows.append(extract_features(image, detected_circle))
            labels.append(label)

    if not feature_rows:
        raise ValueError("No training samples could be built from the provided data")

    X = np.asarray(feature_rows, dtype=np.float32)
    y = np.asarray(labels)
    return X, y


def train_knn(X: np.ndarray, y: np.ndarray, n_neighbors: int = 5):
    """
    Train a K-Nearest Neighbors classifier.

    Returns:
        Trained KNN model.
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X, y)
    return model


def train_svm(X: np.ndarray, y: np.ndarray):
    """
    Train a Support Vector Machine classifier.

    Returns:
        Trained SVM model.
    """
    model = SVC(kernel="rbf", gamma="scale", probability=True)
    model.fit(X, y)
    return model


def save_model(model, scaler, model_path: str, scaler_path: str) -> None:
    """
    Save the trained model and scaler to disk using pickle/joblib.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    if scaler is not None:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)


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
    X, y = load_training_data(labels_path, images_dir)
    X_scaled, scaler = normalize_features(X, scaler=None)

    if model_type.lower() == "knn":
        model = train_knn(X_scaled, y)
    elif model_type.lower() == "svm":
        model = train_svm(X_scaled, y)
    else:
        raise ValueError("model_type must be 'knn' or 'svm'")

    save_model(model, scaler, "models/classifier.pkl", "models/scaler.pkl")
