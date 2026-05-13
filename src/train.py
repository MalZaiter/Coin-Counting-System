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
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.detect import detect_coins
from src.features import extract_features
from src.utils import load_image, normalize_features



def load_training_data(labels_path: str, images_dir: str, return_groups: bool = False) -> tuple:
    """
    Load images and labels for training from YOLO format (.txt files) or CSV.
    
    Supports:
    - YOLO format: data/training/labels/*.txt files (class_id x_norm y_norm w_norm h_norm)
    - CSV format: columns: image_path, x, y, radius, coin_type
    
    Args:
        labels_path: Path to labels directory (for YOLO) or CSV file
        images_dir: Directory containing training images

    Returns:
        (X, y) — feature matrix and label vector
        or (X, y, groups) if return_groups=True
    """
    labels_path = Path(labels_path)
    images_dir = Path(images_dir)
    
    # Class mapping for YOLO format
    CLASS_MAP = {
        0: "1cent", 1: "2cent", 2: "5cent", 3: "10cent",
        4: "20cent", 5: "50cent", 6: "1euro", 7: "2euro",
    }
    
    feature_rows = []
    label_list = []
    groups = []

    # Try YOLO format first (directory with .txt files)
    if labels_path.is_dir():
        label_files = sorted(labels_path.glob("*.txt"))
        if not label_files:
            raise FileNotFoundError(f"No .txt label files found in {labels_path}")
        
        for label_file in label_files:
            image_name = label_file.stem + ".jpg"
            image_path = images_dir / image_name
            if not image_path.exists():
                image_path = images_dir / "images" / image_name
            if not image_path.exists():
                continue
            
            image = load_image(str(image_path))
            if image is None:
                continue
            
            h, w = image.shape[:2]
            
            # Read YOLO format: class_id x_norm y_norm w_norm h_norm
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_norm = float(parts[1])
                    y_norm = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])
                    
                    # Convert to pixel coordinates (bounding box format)
                    x_center = x_norm * w
                    y_center = y_norm * h
                    w_px = w_norm * w
                    h_px = h_norm * h
                    x1 = int(x_center - w_px / 2)
                    y1 = int(y_center - h_px / 2)
                    x2 = int(x_center + w_px / 2)
                    y2 = int(y_center + h_px / 2)
                    
                    # Ensure bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w - 1, x2)
                    y2 = min(h - 1, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    bbox = (x1, y1, x2, y2)
                    label = CLASS_MAP.get(class_id, f"class_{class_id}")
                    
                    feature_rows.append(extract_features(image, bbox))
                    label_list.append(label)
                    groups.append(label_file.stem)
    
    # Fall back to CSV format
    elif labels_path.is_file() and labels_path.suffix == ".csv":
        with labels_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_ref = row.get("image_path") or row.get("image")
                if not image_ref:
                    continue
                
                filename = Path(image_ref).name
                image_path = images_dir / filename
                if not image_path.exists():
                    image_path = images_dir / "images" / filename
                if not image_path.exists():
                    continue
                
                label = row.get("coin_type") or row.get("label")
                if not label:
                    continue
                
                image = load_image(str(image_path))
                if image is None:
                    continue
                
                # CSV has x, y, radius format
                x = int(float(row["x"]))
                y = int(float(row["y"]))
                radius = int(float(row["radius"]))
                circle = (x, y, radius)
                
                feature_rows.append(extract_features(image, circle))
                label_list.append(str(label).strip())
                groups.append(filename)
    else:
        raise FileNotFoundError(f"Labels must be a directory (YOLO) or CSV file: {labels_path}")

    if not feature_rows:
        raise ValueError("No training samples could be built from the provided data")

    X = np.asarray(feature_rows, dtype=np.float32)
    y = np.asarray(label_list)
    g = np.asarray(groups)
    
    if return_groups:
        return X, y, g
    return X, y




def train_knn(X: np.ndarray, y: np.ndarray, n_neighbors: int = 5):
    """Train a K-Nearest Neighbors classifier."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X, y)
    return model


def train_svm(X: np.ndarray, y: np.ndarray):
    """Train a Support Vector Machine classifier."""
    model = SVC(kernel="rbf", gamma="auto", C=10,class_weight="balanced", probability=True)
    model.fit(X, y)
    return model


def save_model(model, scaler, model_path: str, scaler_path: str) -> None:
    """Save the trained model and scaler to disk."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    if scaler is not None:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)



def train(labels_path: str, images_dir: str, model_type: str = "svm", test_size: float = 0.2) -> dict:
    """
    Full training pipeline: load data, split, scale, train, evaluate, and save.

    Args:
        labels_path: Path to labels CSV file
        images_dir: Path to training images directory
        model_type: 'knn' or 'svm'
        test_size: Fraction for test set (default 0.2)

    Returns:
        Dictionary with training metrics
    """
    print("\n" + "="*70)
    print("TRAINING COIN CLASSIFIER")
    print("="*70)

    print("\nLoading data...")
    X, y, groups = load_training_data(labels_path, images_dir, return_groups=True)
    print(f"  Total samples: {len(X)}")
    print(f"  Classes: {np.unique(y)}")
    print(f"  Unique source images: {len(np.unique(groups))}")

    print(f"\nSplitting by image groups ({100*(1-test_size):.0f}% train, {100*test_size:.0f}% test)...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_groups = set(groups[train_idx].tolist())
    test_groups = set(groups[test_idx].tolist())
    overlap = len(train_groups.intersection(test_groups))
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Train images: {len(train_groups)} | Test images: {len(test_groups)} | Overlap: {overlap}")

    print("\nScaling features...")
    X_train_scaled, scaler = normalize_features(X_train, scaler=None)
    X_test_scaled, _ = normalize_features(X_test, scaler=scaler)

    print(f"\nTraining {model_type.upper()} classifier...")
    if model_type.lower() == "knn":
        model = train_knn(X_train_scaled, y_train)
    elif model_type.lower() == "svm":
        model = train_svm(X_train_scaled, y_train)
    else:
        raise ValueError("model_type must be 'knn' or 'svm'")

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


