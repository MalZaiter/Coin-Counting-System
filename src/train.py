"""
Training module for the coin classification pipeline.

Usage
-----
  python src/train.py --data_dir data/train --labels data/labels/labels.csv

Labels CSV format
-----------------
  filename,label
  img001.jpg,quarter
  img002.jpg,dime
  ...

Each image in data_dir should already contain a single, pre-cropped coin ROI,
OR the script can load full images + their detected circles from a JSON manifest
(see --manifest flag).

Outputs
-------
  models/coin_classifier.pkl   — trained SVM (default) or KNN
  models/scaler.pkl            — fitted StandardScaler
"""

import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

try:
    from .features import extract_features, extract_features_batch
except ImportError:
    from features import extract_features, extract_features_batch


# Constants

MODELS_DIR = "models"
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "coin_classifier.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")


# Data loading helpers

def load_from_csv(data_dir, labels_csv):
    """
    Load pre-cropped ROI images and their labels from a CSV file.

    Each row: filename,label
    The image is loaded as-is (should be a tight crop of one coin).
    A synthetic circle spanning the full image is used for feature extraction.

    Returns (X, y_str) where X is a list of feature vectors and y_str is a
    list of string labels.
    """
    import csv

    X, y_str = [], []

    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = os.path.join(data_dir, row["filename"].strip())
            label = row["label"].strip()

            image = cv2.imread(img_path)
            if image is None:
                print(f"[WARN] Cannot read {img_path}, skipping.")
                continue

            h, w = image.shape[:2]
            cx, cy = w // 2, h // 2
            r = min(cx, cy)
            circle = (cx, cy, r)

            fv = extract_features(image, circle)
            if fv is None:
                print(f"[WARN] Feature extraction failed for {img_path}, skipping.")
                continue

            X.append(fv)
            y_str.append(label)

    return X, y_str


def load_from_manifest(manifest_path):
    """
    Load full images with pre-detected circles from a JSON manifest.

    Manifest format:
    [
      {
        "image": "data/train/scene01.jpg",
        "circles": [[cx, cy, r], ...],
        "labels": ["quarter", "dime", ...]
      },
      ...
    ]

    Returns (X, y_str).
    """
    with open(manifest_path) as f:
        entries = json.load(f)

    X, y_str = [], []

    for entry in entries:
        image = cv2.imread(entry["image"])
        if image is None:
            print(f"[WARN] Cannot read {entry['image']}, skipping.")
            continue

        circles = entry["circles"]
        labels = entry["labels"]

        if len(circles) != len(labels):
            print(f"[WARN] Circle/label mismatch in {entry['image']}, skipping.")
            continue

        for circle, label in zip(circles, labels):
            fv = extract_features(image, circle)
            if fv is None:
                continue
            X.append(fv)
            y_str.append(label)

    return X, y_str


# Model builders

def build_knn(n_neighbors=5):
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights="distance",
        metric="euclidean",
    )


def build_svm():
    return SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
    )


# Training

# Scale features, encode labels, train classifier, and evaluate.
def train(X_raw, y_str, model_type="svm", test_size=0.2, random_state=42):
    if len(X_raw) == 0:
        raise ValueError("No training samples found — check your data directory and labels file.")

    X = np.vstack(X_raw)
    classes = sorted(set(y_str))
    print(f"[INFO] Classes: {classes}")
    print(f"[INFO] Total samples: {len(y_str)}")

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == "knn":
        clf = build_knn()
        print("[INFO] Training KNN classifier …")
    else:
        clf = build_svm()
        print("[INFO] Training SVM classifier …")

    if test_size > 0 and len(X_scaled) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        print("--- Confusion Matrix ---")
        print(confusion_matrix(y_test, y_pred))
        print()
    else:
        print("[INFO] Dataset too small for a held-out split — training on all data.")
        clf.fit(X_scaled, y)

    cv_scores = cross_val_score(clf, X_scaled, y, cv=min(5, len(X_scaled)), scoring="accuracy")
    print(f"[INFO] Cross-val accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return clf, scaler, le


# Persistence

def save_models(clf, scaler, le):
    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"[INFO] Classifier saved → {CLASSIFIER_PATH}")

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[INFO] Scaler saved → {SCALER_PATH}")

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"[INFO] Label encoder saved → {LABEL_ENCODER_PATH}")


def load_models():
    """
    Load saved classifier, scaler, and label encoder.
    Returns (clf, scaler, le) — used by predict.py.
    """
    for path in [CLASSIFIER_PATH, SCALER_PATH, LABEL_ENCODER_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\nRun training first."
            )

    with open(CLASSIFIER_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    return clf, scaler, le


def predict_single(image, circle, clf, scaler, le):
    """
    Convenience wrapper used by predict.py and main.py.

    Returns (label_str, confidence) or (None, None) on failure.
    """
    fv = extract_features(image, circle)
    if fv is None:
        return None, None

    fv_scaled = scaler.transform(fv.reshape(1, -1))

    label_idx = clf.predict(fv_scaled)[0]
    label_str = le.inverse_transform([label_idx])[0]

    if hasattr(clf, "predict_proba"):
        confidence = clf.predict_proba(fv_scaled).max()
    else:
        confidence = None

    return label_str, confidence


# CLI entry point

def parse_args():
    parser = argparse.ArgumentParser(description="Train coin classifier")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--labels", metavar="CSV",
        help="CSV file with filename,label columns (use with --data_dir)"
    )
    source.add_argument(
        "--manifest", metavar="JSON",
        help="JSON manifest with image paths + circles + labels"
    )

    parser.add_argument(
        "--data_dir", default="data/train",
        help="Directory containing training images (used with --labels)"
    )
    parser.add_argument(
        "--model", choices=["svm", "knn"], default="svm",
        help="Classifier to train (default: svm)"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Fraction of data to hold out for evaluation (default: 0.2)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.manifest:
        print(f"[INFO] Loading data from manifest: {args.manifest}")
        X, y_str = load_from_manifest(args.manifest)
    else:
        print(f"[INFO] Loading data from {args.data_dir} using labels {args.labels}")
        X, y_str = load_from_csv(args.data_dir, args.labels)

    clf, scaler, le = train(X, y_str, model_type=args.model, test_size=args.test_size)
    save_models(clf, scaler, le)
    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
