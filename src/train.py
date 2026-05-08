"""
train.py — Model Training
Person 3: Feature Extraction & Training

Responsibilities:
- Load labeled training data
- Split into train/test sets
- Train KNN or SVM classifier
- Evaluate model performance
- Save trained model and scaler
"""

import csv
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.detect import detect_coins
from src.features import extract_features
from src.utils import load_image, normalize_features


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_LABELS = ROOT / "archive data" / "labels.csv"
ARCHIVE_IMAGES = ROOT / "archive data"


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

    # Support the renamed archive folder that currently exists in this project.
    if image_ref.startswith(("archive\\images", "archive/images")):
        archive_alias = Path("archive data") / "images" / candidate.name

        alias_relative_to_cwd = Path.cwd() / archive_alias
        if alias_relative_to_cwd.exists():
            return str(alias_relative_to_cwd)

        alias_relative_to_images = Path(images_dir) / archive_alias
        if alias_relative_to_images.exists():
            return str(alias_relative_to_images)

        alias_relative_to_root = ROOT / archive_alias
        if alias_relative_to_root.exists():
            return str(alias_relative_to_root)

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


def train(labels_path: str, images_dir: str, model_type: str = "svm", test_size: float = 0.2) -> dict:
    """
    Full training pipeline:
    1. Load data
    2. Split into train/test sets
    3. Scale features
    4. Train classifier
    5. Evaluate on test set
    6. Save model

    Args:
        labels_path: Path to labels file.
        images_dir:  Path to training images directory.
        model_type:  'knn' or 'svm'.
        test_size:   Fraction for test set (default 0.2 = 80/20 split).
        
    Returns:
        Dictionary with training results and metrics.
    """
    print("\n" + "="*70)
    print("TRAINING COIN CLASSIFIER")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    X, y = load_training_data(labels_path, images_dir)
    print(f"  Total samples: {len(X)}")
    print(f"  Classes: {np.unique(y)}")
    
    # Split data
    print(f"\nSplitting data ({100*(1-test_size):.0f}% train, {100*test_size:.0f}% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Scale features
    print("\nScaling features...")
    X_train_scaled, scaler = normalize_features(X_train, scaler=None)
    X_test_scaled, _ = normalize_features(X_test, scaler=scaler)
    
    # Train model
    print(f"\nTraining {model_type.upper()} classifier...")
    if model_type.lower() == "knn":
        model = train_knn(X_train_scaled, y_train)
    elif model_type.lower() == "svm":
        model = train_svm(X_train_scaled, y_train)
    else:
        raise ValueError("model_type must be 'knn' or 'svm'")
    
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Save model
    print("\nSaving model...")
    save_model(model, scaler, "models/classifier.pkl", "models/scaler.pkl")
    print(f"  Model saved to: models/classifier.pkl")
    print(f"  Scaler saved to: models/scaler.pkl")
    
    results = {
        "model_type": model_type,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "classes": list(np.unique(y)),
    }
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70 + "\n")
    
    return results


def main():
    """Main training script."""
    import sys
    
    labels_path = str(ARCHIVE_LABELS)
    images_dir = str(ARCHIVE_IMAGES)
    model_type = "svm"
    
    # Allow override via command line
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    
    try:
        results = train(labels_path, images_dir, model_type=model_type)
        print("Training successful!")
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
