#!/usr/bin/env python3
"""Train classifier and test end-to-end coin type classification.

This script:
1) Trains the model using archive labels
2) Loads the trained model/scaler
3) Detects coins in complex test images
4) Extracts features and predicts a type for each detected coin
5) Saves annotated images and a CSV summary of predictions

Run with:
    python tester/test_train.py
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import joblib
import numpy as np

from src.detect import detect_coins
from src.features import extract_features
from src.train import ARCHIVE_IMAGES, ARCHIVE_LABELS, train
from src.utils import list_images


ROOT = Path(__file__).resolve().parents[1]
COMPLEX_TESTS_DIR = ROOT / "archive data" / "complex_tests"
OUT_DIR = ROOT / "outputs" / "train_classification"
CSV_PATH = OUT_DIR / "predictions.csv"

MODEL_PATH = ROOT / "models" / "classifier.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
LOG_PATH = ROOT / "outputs" / "train_classification" / "test_train_output.txt"

MODEL_TYPE = "svm"
MAX_IMAGES = None  # Set an int (e.g., 10) for a faster smoke test.


COIN_VALUE_MAP = {
    "1cent": 0.01,
    "2cent": 0.02,
    "5cent": 0.05,
    "10cent": 0.10,
    "20cent": 0.20,
    "50cent": 0.50,
    "1euro": 1.00,
    "2euro": 2.00,
}


def print_log(text: str, file_handle=None) -> None:
    """Print to both terminal and log file."""
    print(text)
    if file_handle is not None:
        file_handle.write(text + "\n")
        file_handle.flush()


def ensure_output_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def annotate_predictions(image: np.ndarray, circles: list[tuple[int, int, int]], labels: list[str]) -> np.ndarray:
    """Draw detected circles and predicted labels on the image."""
    annotated = image.copy()

    for (x, y, r), label in zip(circles, labels):
        x, y, r = int(x), int(y), int(r)
        cv2.circle(annotated, (x, y), r, (0, 255, 0), 2)
        cv2.circle(annotated, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            label,
            (x - r, max(20, y - r - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def _predict_one(model, scaler, feature_vector: np.ndarray) -> str:
    fv2d = feature_vector.reshape(1, -1)
    if scaler is not None:
        fv2d = scaler.transform(fv2d)
    return str(model.predict(fv2d)[0])


def run_classification() -> None:
    ensure_output_dirs()

    if not COMPLEX_TESTS_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {COMPLEX_TESTS_DIR}")

    # Open log file for writing
    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        print_log("\n" + "=" * 70, log_file)
        print_log("TRAIN + CLASSIFICATION TEST", log_file)
        print_log("=" * 70, log_file)

        print_log("\n[1/3] Training model...", log_file)
        results = train(str(ARCHIVE_LABELS), str(ARCHIVE_IMAGES), model_type=MODEL_TYPE)

        print_log("\n[2/3] Loading trained model artifacts...", log_file)
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            raise FileNotFoundError("Expected trained model artifacts were not found under models/")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        print_log("\n[3/3] Classifying detected coins in complex test images...", log_file)
        image_paths = list_images(str(COMPLEX_TESTS_DIR))
        if MAX_IMAGES is not None:
            image_paths = image_paths[:MAX_IMAGES]

        rows: list[dict] = []
        total_detected = 0
        all_label_counts = Counter()

        for image_path in image_paths:
            image_path = str(image_path)
            image_name = Path(image_path).name
            image = cv2.imread(image_path)
            if image is None:
                print_log(f"  [E] Failed to load: {image_name}", log_file)
                continue

            circles = detect_coins(image)
            labels = []

            for (x, y, r) in circles:
                fv = extract_features(image, (x, y, r)).astype(np.float32)
                label = _predict_one(model, scaler, fv)
                labels.append(label)

                rows.append(
                    {
                        "image": image_name,
                        "x": int(x),
                        "y": int(y),
                        "radius": int(r),
                        "predicted_label": label,
                        "predicted_value_eur": COIN_VALUE_MAP.get(label, 0.0),
                    }
                )

            total_detected += len(circles)
            label_counts = Counter(labels)
            all_label_counts.update(label_counts)

            total_value = sum(COIN_VALUE_MAP.get(lbl, 0.0) for lbl in labels)
            print_log(f"  [OK] {image_name}: {len(circles)} detected -> {dict(label_counts)} | value={total_value:.2f} EUR", log_file)

            annotated = annotate_predictions(image, circles, labels)
            out_img = OUT_DIR / f"{Path(image_name).stem}_classified.jpg"
            cv2.imwrite(str(out_img), annotated)

        with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
            fieldnames = ["image", "x", "y", "radius", "predicted_label", "predicted_value_eur"]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        grand_total = sum(COIN_VALUE_MAP.get(lbl, 0.0) * count for lbl, count in all_label_counts.items())

        print_log("\n" + "=" * 70, log_file)
        print_log("RESULT SUMMARY", log_file)
        print_log("=" * 70, log_file)
        print_log(f"Model type: {results['model_type']}", log_file)
        print_log(f"Train/Test Accuracy: {results['accuracy']:.4f}", log_file)
        print_log(f"Images processed: {len(image_paths)}", log_file)
        print_log(f"Detected coins: {total_detected}", log_file)
        print_log(f"Predicted class counts: {dict(all_label_counts)}", log_file)
        print_log(f"Total predicted value: {grand_total:.2f} EUR", log_file)
        print_log(f"Annotated outputs: {OUT_DIR}", log_file)
        print_log(f"Prediction CSV: {CSV_PATH}", log_file)
        print_log(f"Log file: {LOG_PATH}", log_file)
        print_log("=" * 70 + "\n", log_file)


if __name__ == "__main__":
    run_classification()
