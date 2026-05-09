#!/usr/bin/env python3
"""Evaluate classifier on labeled circles (ground-truth) and save diagnostics.

This script:
- Loads the trained model and scaler from models/
- Reads archive data/labels.csv (circle-level entries)
- For each labeled circle, extracts features at the labelled coords, predicts
  using the saved model, and records true vs predicted
- Saves a CSV of per-sample results and writes annotated images for
  misclassified samples under outputs/eval/misclassified
"""

from pathlib import Path
import csv
from collections import Counter
import joblib
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from src.train import ARCHIVE_LABELS, ARCHIVE_IMAGES, _resolve_image_path
from src.features import extract_features

# Paths
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MIS_DIR = OUT_DIR / "misclassified"
MIS_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT = OUT_DIR / "eval_results.csv"
REPORT_OUT = OUT_DIR / "classification_report.txt"

MODEL_PATH = ROOT / "models" / "classifier.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"


def annotate_gt_pred(image, x, y, r, true_label, pred_label):
    img = image.copy()
    x, y, r = int(x), int(y), int(r)
    # ground-truth circle in cyan
    cv2.circle(img, (x, y), r, (255, 255, 0), 2)
    # center
    cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    # labels
    cv2.putText(img, f"GT: {true_label}", (x - r, max(16, y - r - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(img, f"P: {pred_label}", (x - r, max(36, y - r - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
    return img


def main():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Model or scaler not found under models/; run training first.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Read labels manifest (CSV)
    rows = []
    with open(ARCHIVE_LABELS, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)

    results = []
    true_labels = []
    pred_labels = []

    for row in rows:
        # Only evaluate circle-level rows (require x,y,radius)
        if not row.get("x") or not row.get("y") or not row.get("radius"):
            continue
        image_ref = row.get("image_path") or row.get("image") or row.get("filename")
        if not image_ref:
            continue

        try:
            image_path = _resolve_image_path(str(ARCHIVE_IMAGES), image_ref)
        except Exception as e:
            print(f"Skipping {image_ref}: {e}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load {image_path}")
            continue

        x = int(float(row["x"]))
        y = int(float(row["y"]))
        r = int(float(row["radius"]))
        true = (row.get("coin_type") or row.get("label") or row.get("class") or "").strip()

        fv = extract_features(image, (x, y, r)).astype(np.float32)
        fv2d = fv.reshape(1, -1)
        if scaler is not None:
            fv2d = scaler.transform(fv2d)

        pred = str(model.predict(fv2d)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = float(np.max(model.predict_proba(fv2d)))
            except Exception:
                prob = None

        results.append({
            "image": Path(image_path).name,
            "x": x,
            "y": y,
            "radius": r,
            "true_label": true,
            "pred_label": pred,
            "prob": prob if prob is not None else "",
        })

        true_labels.append(true)
        pred_labels.append(pred)

        # Save misclassified annotated image
        if true != pred:
            ann = annotate_gt_pred(image, x, y, r, true, pred)
            outp = MIS_DIR / f"{Path(image_path).stem}_x{x}_y{y}.jpg"
            cv2.imwrite(str(outp), ann)

    # Write CSV
    with open(CSV_OUT, "w", encoding="utf-8", newline="") as fh:
        fieldnames = ["image", "x", "y", "radius", "true_label", "pred_label", "prob"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Metrics
    acc = accuracy_score(true_labels, pred_labels) if true_labels else 0.0
    cm = confusion_matrix(true_labels, pred_labels, labels=sorted(list(set(true_labels)))) if true_labels else None
    report = classification_report(true_labels, pred_labels, zero_division=0) if true_labels else ""

    with open(REPORT_OUT, "w", encoding="utf-8") as fh:
        fh.write(f"Accuracy: {acc:.4f}\n\n")
        fh.write("Classification Report:\n")
        fh.write(report)
        fh.write("\n\n")
        if cm is not None:
            fh.write("Confusion Matrix (rows=true, cols=pred)\n")
            labels = sorted(list(set(true_labels)))
            fh.write("," + ",".join(labels) + "\n")
            for i, lab in enumerate(labels):
                fh.write(lab + "," + ",".join(str(x) for x in cm[i].tolist()) + "\n")

    print("Evaluation complete")
    print(f"Results CSV: {CSV_OUT}")
    print(f"Misclassified images: {MIS_DIR} (examples saved)")
    print(f"Report: {REPORT_OUT}")


if __name__ == "__main__":
    main()
