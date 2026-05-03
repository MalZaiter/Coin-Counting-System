"""
main.py — Full Pipeline Runner
Person 4: Integration & Evaluation

Usage:
    python main.py --mode train   --images data/train  --labels data/labels/train.csv
    python main.py --mode predict --image  data/test/sample.jpg
    python main.py --mode evaluate --images data/test  --labels data/labels/test.csv
"""

import argparse

from src.preprocess import preprocess
from src.detect import detect_coins
from src.features import extract_features
from src.train import train
from src.predict import predict
from src.evaluate import evaluate


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    pass


def run_train(args: argparse.Namespace) -> None:
    """Run the training pipeline."""
    pass


def run_predict(args: argparse.Namespace) -> None:
    """Run inference on a single image and display results."""
    pass


def run_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation on a test set and print metrics."""
    pass


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
