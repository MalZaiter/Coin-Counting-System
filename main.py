"""
main.py — Full Pipeline Runner
Person 4: Integration & Evaluation

Usage:
    python main.py --mode train   --images data/train  --labels data/labels/train.csv
    python main.py --mode predict --image  data/test/sample.jpg
    python main.py --mode evaluate --images data/test  --labels data/labels/test.csv
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

import argparse

from src.preprocess import preprocess
from src.detect import detect_coins
from src.features import extract_features
from src.train import train
from src.predict import predict
from src.evaluate import evaluate


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Coin Detection & Classification Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict", "evaluate"],
        required=True,
        help="Pipeline mode: train, predict, or evaluate"
    )
    parser.add_argument(
        "--images",
        type=str,
        help="Path to training/test images directory"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to single image for prediction"
    )
    parser.add_argument(
        "--labels",
        type=str,
        help="Path to labels CSV or directory with label files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/coin_classifier.pkl",
        help="Path to trained model (default: models/coin_classifier.pkl)"
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default=None,
        help="Path to trained scaler (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/",
        help="Output directory for results"
    )
    return parser.parse_args()


def run_train(args: argparse.Namespace) -> None:
    """Run the training pipeline."""
    if not args.images or not args.labels:
        print("Error: --images and --labels required for train mode")
        return

    print(f"\n{'='*60}")
    print("TRAINING PIPELINE")
    print(f"{'='*60}")
    print(f"Images directory: {args.images}")
    print(f"Labels file: {args.labels}")
    print(f"Output model: {args.model}")
    print(f"{'='*60}\n")

    try:
        train(args.images, args.labels, args.model, args.scaler)
        print("\n✓ Training complete!")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")


def run_predict(args: argparse.Namespace) -> None:
    """Run inference on a single image and display results."""
    if not args.image:
        print("Error: --image required for predict mode")
        return

    print(f"\n{'='*60}")
    print("PREDICTION PIPELINE")
    print(f"{'='*60}")
    print(f"Image: {args.image}")
    print(f"Model: {args.model}")
    print(f"{'='*60}\n")

    try:
        import cv2
        import os

        result = predict(args.image, args.model, args.scaler)

        # Display results
        print(f"Coins detected: {len(result['coins'])}")
        print(f"Coin counts: {result['label_counts']}")
        print(f"Total value: ${result['total_value']:.2f}")

        # Save annotated image
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, "prediction_result.jpg")
        cv2.imwrite(output_path, result['annotated_image'])
        print(f"\n✓ Result saved to {output_path}")

        # Display image
        cv2.imshow("Coin Detection Result", result['annotated_image'])
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"\n✗ Prediction failed: {e}")


def run_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation on a test set and print metrics."""
    if not args.images or not args.labels:
        print("Error: --images and --labels required for evaluate mode")
        return

    print(f"\n{'='*60}")
    print("EVALUATION PIPELINE")
    print(f"{'='*60}")
    print(f"Test images directory: {args.images}")
    print(f"Labels file: {args.labels}")
    print(f"Model: {args.model}")
    print(f"{'='*60}\n")

    try:
        result = evaluate(args.images, args.labels, args.model, args.scaler)
        print(f"\n✓ Evaluation complete!")
        print(f"Accuracy: {result['accuracy']:.4f}")
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")


def main() -> None:
    args = parse_args()

    if args.mode == "train":
        run_train(args)
    elif args.mode == "predict":
        run_predict(args)
    elif args.mode == "evaluate":
        run_evaluate(args)
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
