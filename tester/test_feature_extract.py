#!/usr/bin/env python3
"""Test, annotate, and export feature extraction for raw and complex datasets.

Detects circles on images, extracts features, annotates images with circles,
validates feature vectors, exports to CSV, and prints sample outputs.

Run with: python -m tester.test_feature_extract
"""

from pathlib import Path
import csv
import sys

import cv2
import numpy as np

from src.utils import list_images, load_image
from src.detect import detect_coins
from src.features import extract_features


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "features_extracted"
FOLDERS = [("raw", ROOT / "archive data" / "raw"), ("complex_tests", ROOT / "archive data" / "complex_tests")]

EXPECTED_FEATURE_LEN = 59


def ensure_out():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)


def annotate_image(image, circles):
    """Draw detected circles on image.
    
    Args:
        image: BGR image array
        circles: List of (x, y, radius) tuples
        
    Returns:
        Annotated image with circles drawn
    """
    annotated = image.copy()
    
    for i, (x, y, radius) in enumerate(circles, 1):
        x, y = int(x), int(y)
        radius = int(radius)
        
        # Draw circle outline in green
        cv2.circle(annotated, (x, y), radius, (0, 255, 0), 2)
        
        # Draw center point in red
        cv2.circle(annotated, (x, y), 3, (0, 0, 255), -1)
        
        # Draw circle number and radius label
        label = f"#{i} r={radius}"
        cv2.putText(annotated, label, (x - 30, y - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return annotated


def test_and_export_folder(name: str, folder: Path, max_images: int = None):
    """Extract features, annotate images, validate, and export.
    
    Args:
        name: Folder name (e.g., 'raw', 'complex_tests')
        folder: Path to image folder
        max_images: Max images to process (None = all)
        
    Returns:
        Tuple: (processed, detected_count, samples, skipped)
    """
    images = list_images(str(folder))
    if max_images:
        images = images[:max_images]
    
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = OUT_ROOT / f"{name}.csv"
    
    rows = []
    samples = []
    processed = 0
    skipped = 0
    errors = []
    
    print(f"\n{'='*60}")
    print(f"Processing {name} folder: {folder}")
    print(f"{'='*60}")
    
    for img_path in images:
        try:
            image = load_image(str(img_path))
            if image is None:
                print(f"  [X] {Path(img_path).name} - failed to load")
                skipped += 1
                continue
            
            img_p = Path(img_path)
            circles = detect_coins(image)
            
            # Extract features and validate
            for i, (x, y, r) in enumerate(circles):
                fv = extract_features(image, (x, y, r)).astype(float)
                
                # validations
                if fv.ndim != 1 or fv.shape[0] != EXPECTED_FEATURE_LEN:
                    raise RuntimeError(f"Feature length mismatch for {img_p} at {(x,y,r)}: got {fv.shape[0]}")
                if np.isnan(fv).any() or np.isinf(fv).any():
                    raise RuntimeError(f"Invalid numeric values in features for {img_p} at {(x,y,r)}")
                
                row = {"image": str(img_p), "x": int(x), "y": int(y), "radius": int(r)}
                row.update({f"f{i}": float(fv[i]) for i in range(len(fv))})
                rows.append(row)
                
                # keep up to 5 sample detections for console print
                if len(samples) < 5:
                    samples.append({"image": str(img_p), "x": int(x), "y": int(y), "radius": int(r), "features": fv.tolist()})
            
            # Annotate image with circles
            if not circles:
                # Still save the image even if no circles detected
                out_path = out_dir / (img_p.stem + "_annotated.jpg")
                cv2.imwrite(str(out_path), image)
                print(f"  [-] {img_p.name} - no circles detected")
            else:
                annotated = annotate_image(image, circles)
                out_path = out_dir / (img_p.stem + "_annotated.jpg")
                cv2.imwrite(str(out_path), annotated)
                print(f"  [OK] {img_p.name} - {len(circles)} circle(s)")
            
            processed += 1
            
        except Exception as e:
            errors.append((Path(img_path).name, str(e)))
            print(f"  [E] {Path(img_path).name} - ERROR: {e}")
            skipped += 1
    
    # Write aggregate CSV
    if rows:
        feature_count = max(len(r) - 4 for r in rows)
        fieldnames = ["image", "x", "y", "radius"] + [f"f{i}" for i in range(feature_count)]
        with csv_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                # ensure keys present
                for i in range(feature_count):
                    if f"f{i}" not in r:
                        r[f"f{i}"] = ""
                writer.writerow(r)
    
    print(f"\n{name.upper()} Summary:")
    print(f"  Processed: {processed}")
    print(f"  Detections: {len(rows)}")
    print(f"  Skipped: {skipped}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for fname, err in errors:
            print(f"    - {fname}: {err}")
    
    return processed, len(rows), samples, skipped


def main():
    """Test, extract, and annotate features for all folders."""
    print("\n" + "="*60)
    print("TESTING FEATURE EXTRACTION")
    print("="*60)
    
    ensure_out()
    total_processed = 0
    total_detections = 0
    total_skipped = 0
    all_samples = {}
    
    for name, folder in FOLDERS:
        if not folder.exists():
            print(f"[WARN] folder missing: {folder}")
            continue
        processed, detections, samples, skipped = test_and_export_folder(name, folder)
        total_processed += processed
        total_detections += detections
        total_skipped += skipped
        all_samples[name] = samples
    
    # Print final summary and samples
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"  Total Images Processed: {total_processed}")
    print(f"  Total Detections: {total_detections}")
    print(f"  Total Skipped: {total_skipped}")
    print(f"  Annotated Images: {OUT_ROOT}")
    print(f"  CSV Exports: {OUT_ROOT}/raw.csv, {OUT_ROOT}/complex_tests.csv")
    print()
    
    # Print sample feature vectors
    for name, samples in all_samples.items():
        print(f"Sample features from {name} (up to 5):")
        if not samples:
            print("  (no detections)")
            continue
        for s in samples:
            feat_preview = ", ".join(f"{x:.4g}" for x in s["features"][:10])
            print(f"  {Path(s['image']).name} @ ({s['x']},{s['y']},r={s['radius']}): [{feat_preview}, ...]")
        print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
