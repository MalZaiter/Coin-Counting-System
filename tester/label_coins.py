#!/usr/bin/env python3
"""Interactive coin type labeling script.

Displays detected coin circles from images and lets you label each one.
Saves labels to data/labels.csv in format: image_path, x, y, radius, coin_type

Run with: python -m tester.label_coins
"""

from pathlib import Path
import csv
import sys

import cv2
import numpy as np

from src.utils import list_images, load_image, crop_roi
from src.detect import detect_coins


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
LABELS_FILE = ROOT / "data" / "labels.csv"
WINDOW_SIZE = 400  # Display size for cropped circles


def display_circle(image, circle, window_name="Coin"):
    """Display a cropped circle region in a resizable window.
    
    Args:
        image: Original image
        circle: (x, y, radius) tuple
        window_name: Name of display window
    """
    x, y, radius = circle
    
    # Crop region around circle (with padding)
    padding = int(radius * 0.5)
    x1 = max(0, int(x - radius - padding))
    y1 = max(0, int(y - radius - padding))
    x2 = min(image.shape[1], int(x + radius + padding))
    y2 = min(image.shape[0], int(y + radius + padding))
    
    cropped = image[y1:y2, x1:x2].copy()
    
    # Draw circle on cropped region (adjusted for crop offset)
    adj_x = int(x - x1)
    adj_y = int(y - y1)
    cv2.circle(cropped, (adj_x, adj_y), int(radius), (0, 255, 0), 2)
    cv2.circle(cropped, (adj_x, adj_y), 3, (0, 0, 255), -1)
    
    # Resize for display
    h, w = cropped.shape[:2]
    scale = WINDOW_SIZE / max(h, w)
    display = cv2.resize(cropped, (int(w * scale), int(h * scale)))
    
    # Show image
    cv2.imshow(window_name, display)


def label_image(image_path, circles):
    """Label all circles in an image interactively.
    
    Args:
        image_path: Path to image file
        circles: List of (x, y, radius) tuples
        
    Returns:
        List of (x, y, radius, coin_type) tuples
    """
    if not circles:
        print(f"No circles detected in {Path(image_path).name}")
        return []
    
    image = load_image(str(image_path))
    if image is None:
        print(f"Failed to load {image_path}")
        return []
    
    labeled = []
    
    for i, circle in enumerate(circles, 1):
        x, y, radius = circle
        
        # Display the circle
        display_circle(image, circle, f"Circle {i}/{len(circles)}")
        
        print(f"\n{'='*60}")
        print(f"Image: {Path(image_path).name}")
        print(f"Circle {i}/{len(circles)} - Center: ({int(x)}, {int(y)}), Radius: {int(radius)}")
        print(f"{'='*60}")
        
        while True:
            coin_type = input("Enter coin type (penny/nickel/dime/quarter/unknown) [skip with 's']: ").strip().lower()
            
            if coin_type == 's':
                print("Skipped")
                break
            elif coin_type in ('penny', 'nickel', 'dime', 'quarter', 'unknown'):
                labeled.append((int(x), int(y), int(radius), coin_type))
                print(f"Labeled as: {coin_type}")
                break
            else:
                print("Invalid input. Please enter: penny, nickel, dime, quarter, unknown, or 's' to skip")
        
        cv2.destroyAllWindows()
    
    return labeled


def main():
    """Main labeling loop."""
    print("\n" + "="*60)
    print("COIN TYPE LABELING TOOL")
    print("="*60)
    print(f"Loading images from: {DATA_DIR}")
    print(f"Saving labels to: {LABELS_FILE}")
    print()
    
    images = list_images(str(DATA_DIR))
    
    # Check if labels file already exists
    existing_labels = {}
    if LABELS_FILE.exists():
        with LABELS_FILE.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:  # File has headers
                for row in reader:
                    img = row.get("image_path", "")
                    existing_labels[img] = row
        if existing_labels:
            print(f"Found {len(existing_labels)} existing labels")
            response = input("Continue from where you left off? (y/n): ").strip().lower()
            if response != 'y':
                existing_labels.clear()
                print("Starting fresh...")
    
    all_labels = []
    processed = 0
    skipped = 0
    
    for img_idx, img_path in enumerate(images, 1):
        img_name = Path(img_path).name
        rel_path = Path("data/raw") / img_name
        
        # Skip if already labeled
        if str(rel_path) in existing_labels:
            print(f"[{img_idx}/{len(images)}] {img_name} - Already labeled, skipping")
            skipped += 1
            continue
        
        print(f"\n[{img_idx}/{len(images)}] Processing: {img_name}")
        
        try:
            image = load_image(str(img_path))
            if image is None:
                print(f"Failed to load {img_name}")
                skipped += 1
                continue
            
            circles = detect_coins(image)
            labeled_circles = label_image(img_path, circles)
            
            for x, y, r, coin_type in labeled_circles:
                all_labels.append({
                    "image_path": str(rel_path),
                    "x": x,
                    "y": y,
                    "radius": r,
                    "coin_type": coin_type
                })
            
            processed += 1
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            skipped += 1
        
        # Optional: save progress every N images
        if processed % 10 == 0:
            save_labels(all_labels, existing_labels)
            print(f"\nProgress saved ({processed} images labeled)")
    
    # Final save
    save_labels(all_labels, existing_labels)
    
    print("\n" + "="*60)
    print("LABELING COMPLETE")
    print("="*60)
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Labels created: {len(all_labels)}")
    print(f"Saved to: {LABELS_FILE}")
    print("="*60 + "\n")


def save_labels(new_labels, existing_labels):
    """Save labels to CSV file."""
    # Combine existing + new labels
    combined = {}
    
    # Add existing labels
    for img_path, row in existing_labels.items():
        combined[img_path] = row
    
    # Add new labels
    for label in new_labels:
        key = label["image_path"]
        combined[key] = label
    
    # Write to CSV
    if combined:
        fieldnames = ["image_path", "x", "y", "radius", "coin_type"]
        with LABELS_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in combined.values():
                writer.writerow(row)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nLabeling interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
