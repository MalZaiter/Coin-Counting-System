"""
predict.py — Inference on New Images
Person 4: Integration & Evaluation

Responsibilities:
- Load trained model
- Run full pipeline on new images
- Draw detected coins and labels on output image
- Count total coins and compute total value
"""

import cv2
import numpy as np


def load_model(model_path: str, scaler_path: str = None):
    """
    Load the trained classifier (and optional scaler) from disk.

    Returns:
        (model, scaler) tuple. scaler may be None.
    """
    import joblib

    model = joblib.load(model_path)
    scaler = None
    if scaler_path:
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            scaler = None
    return model, scaler


def predict_coin(feature_vector: np.ndarray, model, scaler=None) -> str:
    """
    Predict the coin type for a single feature vector.

    Returns:
        Predicted label string.
    """

    if feature_vector is None:
        return "unknown"

    fv = np.array(feature_vector)

    if fv.ndim == 1:
        fv = fv.reshape(1, -1)

    # Scale features
    if scaler is not None:
        try:
            fv = scaler.transform(fv)
        except Exception:
            pass

    # Predict label
    prediction = model.predict(fv)[0]

    # Optional confidence check
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(fv)
            max_conf = float(np.max(proba[0]))

            print(f"Prediction confidence: {max_conf:.2f}")

        except Exception:
            pass

    return prediction

def draw_results(image: np.ndarray, coins: list, labels: list) -> np.ndarray:
    """
    Draw detected coin circles and their predicted labels on the image.

    Args:
        image:  Original BGR image.
        coins:  List of (x, y, radius) tuples.
        labels: Corresponding predicted label for each coin.

    Returns:
        Annotated image.
    """
    out = image.copy()
    color = (0, 200, 0)
    text_color = (255, 255, 255)
    for i, c in enumerate(coins):
        try:
            x, y, r = map(int, c)
        except Exception:
            continue
        cv2.circle(out, (x, y), r, color, 2)
        lbl = labels[i] if i < len(labels) else "?"
        text = f"{lbl}"
        cv2.putText(out, text, (x - r, y - r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
    return out


def count_coins(labels: list) -> dict:
    """
    Count occurrences of each coin type.

    Returns:
        Dict mapping coin label -> count (e.g., {'quarter': 3, 'dime': 2}).
    """
    from collections import Counter

    return dict(Counter(labels))


def compute_total_value(label_counts: dict) -> float:
    """
    Compute total monetary value from coin counts.

    Returns:
        Total value in dollars (float).
    """
    # Euro coin values aligned with classifier labels.
    coin_values = {
        '1cent': 0.01,
        '2cent': 0.02,
        '5cent': 0.05,
        '10cent': 0.10,
        '20cent': 0.20,
        '50cent': 0.50,
        '1euro': 1.00,
        '2euro': 2.00
    }
    total = 0.0
    for label, count in label_counts.items():
        val = coin_values.get(label, 0.0)
        total += val * count
    return total


def predict(image_path: str, model_path: str, scaler_path: str = None) -> dict:
    """
    Full prediction pipeline for a single image:
    1. Load and preprocess image
    2. Detect coins
    3. Extract features
    4. Predict coin types
    5. Draw and return results

    Returns:
        Dict with keys: annotated_image, label_counts, total_value
    """
    # Load model
    model, scaler = load_model(model_path, scaler_path)
    # Read image (ensure BGR)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    # Ensure image is BGR (3 channels) for color feature extraction
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    coins = []
    labels = []
    # Use project modules for preprocessing, detection, and feature extraction
    from src.preprocess import preprocess
    from src.detect import detect_coins
    from src.features import extract_features
    # detect_coins performs its own preprocessing internally.
    coins = detect_coins(image)
    # Extract features from original image (not preprocessed)
    for c in coins:
        try:
            fv = extract_features(image, c)
            if fv is not None and len(fv) > 0:
                labels.append(predict_coin(fv, model, scaler))
            else:
                labels.append('unknown')
        except Exception as e:
            print(f"Warning: Error extracting features for coin {c}: {e}")
            labels.append('unknown')

    label_counts = count_coins(labels)
    total_value = compute_total_value(label_counts)
    annotated = draw_results(image, coins, labels)
    return {
        'annotated_image': annotated,
        'label_counts': label_counts,
        'total_value': total_value,
        'coins': coins,
        'labels': labels
    }