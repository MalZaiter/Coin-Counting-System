#!/usr/bin/env python3
"""Convert Kaggle Euro coin dataset (YOLO format) to feature-labeled CSV.

Reads YOLO labels, converts bounding boxes to circles, extracts features,
and appends to data/labels.csv for training.

YOLO format: class_id center_x center_y width height (normalized 0-1)
Classes: 0=1cent, 1=2cent, 2=5cent, 3=10cent, 4=20cent, 5=50cent, 6=1euro, 7=2euro

Run with: python -m tester.convert_yolo_to_csv
"""

from pathlib import Path
import csv
import sys

import numpy as np

from src.utils import list_images, load_image
from src.features import extract_features


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_IMAGES = ROOT / "archive" / "images"
ARCHIVE_LABELS = ROOT / "archive" / "labels"
OUTPUT_CSV = ROOT / "data" / "labels.csv"

# YOLO class ID to coin type mapping
CLASS_NAMES = {
    0: "1cent",
    1: "2cent",
    2: "5cent",
    3: "10cent",
    4: "20cent",
    5: "50cent",
    6: "1euro",
    7: "2euro",
}


def yolo_bbox_to_circle(yolo_line, img_width, img_height):
    """Convert YOLO normalized bbox to circle (x, y, radius).
    
    YOLO format: class_id center_x center_y width height (all normalized 0-1)
    
    Args:
        yolo_line: String line from YOLO label file
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Tuple: (class_id, x, y, radius) in pixels
    """
    parts = yolo_line.strip().split()
    if len(parts) < 5:
        return None
    
    class_id = int(parts[0])
    center_x_norm = float(parts[1])
    center_y_norm = float(parts[2])
    width_norm = float(parts[3])
    height_norm = float(parts[4])
    
    # Denormalize
    center_x = center_x_norm * img_width
    center_y = center_y_norm * img_height
    width = width_norm * img_width
    height = height_norm * img_height
    
    # Convert bbox to circle (use average of width/height)
    radius = (width + height) / 4  # Average half-dimension
    
    return (class_id, int(center_x), int(center_y), int(radius))


def convert_archive():
    """Convert archive YOLO labels to features and append to labels.csv."""
    
    print("\n" + "="*70)
    print("CONVERTING KAGGLE EURO COINS (YOLO -> FEATURES)")
    print("="*70)
    
    if not ARCHIVE_IMAGES.exists() or not ARCHIVE_LABELS.exists():
        print(f"ERROR: Archive folders not found")
        print(f"  Images: {ARCHIVE_IMAGES}")
        print(f"  Labels: {ARCHIVE_LABELS}")
        return
    
    images = sorted(ARCHIVE_IMAGES.glob("*.jpg"))
    processed = 0
    skipped = 0
    errors = []
    new_labels = []
    
    print(f"Found {len(images)} images")
    print(f"Output: {OUTPUT_CSV}\n")
    
    for img_idx, img_path in enumerate(images, 1):
        img_name = img_path.name
        label_path = ARCHIVE_LABELS / (img_path.stem + ".txt")
        
        if not label_path.exists():
            print(f"[{img_idx}/{len(images)}] {img_name} - No labels found, skipping")
            skipped += 1
            continue
        
        try:
            # Load image
            image = load_image(str(img_path))
            if image is None:
                print(f"[{img_idx}/{len(images)}] {img_name} - Failed to load image")
                skipped += 1
                continue
            
            img_h, img_w = image.shape[:2]
            
            # Read YOLO labels
            with label_path.open("r") as f:
                yolo_lines = f.readlines()
            
            if not yolo_lines:
                print(f"[{img_idx}/{len(images)}] {img_name} - No detections in label")
                skipped += 1
                continue
            
            # Process each detection
            circles_detected = 0
            for line in yolo_lines:
                result = yolo_bbox_to_circle(line, img_w, img_h)
                if result is None:
                    continue
                
                class_id, x, y, radius = result
                
                # Extract features
                fv = extract_features(image, (x, y, radius)).astype(float)
                
                # Validation
                if fv.ndim != 1 or fv.shape[0] != 59:
                    raise RuntimeError(f"Feature length mismatch: {fv.shape[0]}")
                if np.isnan(fv).any() or np.isinf(fv).any():
                    raise RuntimeError(f"Invalid numeric values in features")
                
                # Build row
                coin_type = CLASS_NAMES.get(class_id, "unknown")
                rel_path = Path("archive") / "images" / img_name
                
                new_labels.append({
                    "image_path": str(rel_path),
                    "x": x,
                    "y": y,
                    "radius": radius,
                    "coin_type": coin_type,
                })
                circles_detected += 1
            
            print(f"[{img_idx}/{len(images)}] {img_name} - {circles_detected} coins")
            processed += 1
            
        except Exception as e:
            errors.append((img_name, str(e)))
            print(f"[{img_idx}/{len(images)}] {img_name} - ERROR: {e}")
            skipped += 1
    
    # Append to existing labels.csv
    print(f"\n{'='*70}")
    print(f"Appending {len(new_labels)} labels to {OUTPUT_CSV.name}")
    print(f"{'='*70}")
    
    # Read existing labels
    existing_labels = {}
    if OUTPUT_CSV.exists():
        with OUTPUT_CSV.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for row in reader:
                    key = (row.get("image_path"), row.get("x"), row.get("y"), row.get("radius"))
                    existing_labels[key] = row
    
    # Combine and write
    all_labels = list(existing_labels.values()) + new_labels
    
    if all_labels:
        fieldnames = ["image_path", "x", "y", "radius", "coin_type"]
        with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_labels:
                writer.writerow(row)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"CONVERSION COMPLETE")
    print(f"{'='*70}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"New labels: {len(new_labels)}")
    print(f"Total in CSV: {len(all_labels)}")
    if errors:
        print(f"Errors: {len(errors)}")
        for fname, err in errors[:5]:
            print(f"  - {fname}: {err}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        convert_archive()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
