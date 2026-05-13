"""End-to-end runner for the coin-counting pipeline."""

from pathlib import Path
from typing import List

import cv2

from src.detect import detect_coins, visualize_detections
from src.predict import predict
from src.train import train
from src.utils import load_image


ROOT = Path(__file__).resolve().parent
TRAIN_LABELS = ROOT / "data" / "training" / "labels"  # YOLO format directory
TRAIN_IMAGES = ROOT / "data" / "training"
RAW_IMAGES = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "classifier.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"
OUTPUT_DIR = ROOT / "outputs" / "predictions"
DETECTION_DIR = ROOT / "outputs" / "detections"


def _iter_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return sorted(
        p for p in path.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def main() -> int:
    print("TRAINING MODEL")
    train_results = train(str(TRAIN_LABELS), str(TRAIN_IMAGES), model_type="svm")
    print(f"Training accuracy: {train_results['accuracy']:.4f}")

    print("\nVISUALIZING TRAINING DETECTIONS")
    DETECTION_DIR.mkdir(parents=True, exist_ok=True)

    train_images = _iter_images(TRAIN_IMAGES / "images")
    if train_images:
        for image_path in train_images[::]:  # Visualize first 10 training images
            try:
                image = load_image(str(image_path))
                circles = detect_coins(image)
                annotated = visualize_detections(image, circles)
                output_path = DETECTION_DIR / f"{image_path.stem}_detections.jpg"
                cv2.imwrite(str(output_path), annotated)
                print(f"  {image_path.name}: {len(circles)} coins detected")
            except Exception as e:
                print(f"  Error processing {image_path.name}: {e}")

    print("\nRUNNING RAW IMAGE PREDICTIONS")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = _iter_images(RAW_IMAGES)
    if not images:
        print(f"No raw images found in {RAW_IMAGES}")
        return 1

    for image_path in images:
        result = predict(str(image_path), str(MODEL_PATH), str(SCALER_PATH))
        output_path = OUTPUT_DIR / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(output_path), result["annotated_image"])

        counts = result["label_counts"]
        print(f"{image_path.name}: {counts} | total={result['total_value']:.2f} | coins={len(result['coins'])}")
        print(f"  saved -> {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())