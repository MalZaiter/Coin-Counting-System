import cv2
import numpy as np

from src.utils import load_image


COPPER_CLASSES = ("1cent", "2cent", "5cent")
GOLD_CLASSES = ("10cent", "20cent", "50cent")
BIMETAL_CLASSES = ("1euro", "2euro")


def load_model(model_path: str, scaler_path: str = None):

    import joblib

    model = joblib.load(model_path)

    scaler = None

    if scaler_path:
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            scaler = None

    return model, scaler


def predict_coin(
    feature_vector: np.ndarray,
    model,
    scaler=None,
    confidence_threshold: float = 0.20
):

    if feature_vector is None:
        return "unknown", 0.0, [], {}

    fv = np.asarray(feature_vector, dtype=np.float32)

    if fv.ndim == 1:
        fv = fv.reshape(1, -1)

    if scaler is not None:
        fv = scaler.transform(fv)

    if not hasattr(model, "predict_proba"):

        label = str(model.predict(fv)[0])

        return label, 1.0, [(label, 1.0)], {label: 1.0}

    probs = model.predict_proba(fv)[0]

    classes = model.classes_

    top_idx = np.argsort(probs)[-3:][::-1]

    top3 = [
        (str(classes[i]), float(probs[i]))
        for i in top_idx
    ]

    best_label, best_conf = top3[0]

    prob_dict = {
        str(classes[i]): float(probs[i])
        for i in range(len(classes))
    }

    if best_conf < confidence_threshold:
        return "unknown", best_conf, top3, prob_dict

    return best_label, best_conf, top3, prob_dict


def draw_results(image: np.ndarray, coins: list, labels: list):

    out = image.copy()

    color = (0, 200, 0)

    text_color = (255, 255, 255)

    for i, c in enumerate(coins):

        try:
            x, y, r = map(int, c)

        except Exception:
            continue

        # green circle
        cv2.circle(out, (x, y), r, color, 2)

        # red center point
        cv2.circle(out, (x, y), 3, (0, 0, 255), -1)

        lbl = labels[i] if i < len(labels) else "?"

        cv2.putText(
            out,
            lbl,
            (x - r, y - r - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2,
            cv2.LINE_AA
        )

    return out


def count_coins(labels: list):

    from collections import Counter

    return dict(Counter(str(label) for label in labels))


def compute_total_value(label_counts: dict):

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


def predict(
    image_path: str,
    model_path: str,
    scaler_path: str = None,
    confidence_threshold: float = 0.20,
):

    model, scaler = load_model(model_path, scaler_path)

    image = load_image(image_path)

    from src.detect import detect_coins
    from src.features import extract_features

    coins = detect_coins(image)

    labels = []
    confidences = []
    top3_predictions = []

    for c in coins:

        try:

            fv = extract_features(image, c)

            if fv is not None and len(fv) > 0:

                label, confidence, top3, _ = predict_coin(
                    fv,
                    model,
                    scaler,
                    confidence_threshold=confidence_threshold,
                )

                labels.append(label)
                confidences.append(float(confidence))
                top3_predictions.append(top3)

            else:

                labels.append("unknown")
                confidences.append(0.0)
                top3_predictions.append([])

        except Exception as e:

            print(f"Warning: Error extracting features for coin {c}: {e}")

            labels.append("unknown")
            confidences.append(0.0)
            top3_predictions.append([])

    # TEMPORARILY DISABLED
    # labels = _refine_labels_by_family_size(
    #     coins,
    #     labels,
    #     probability_dicts,
    #     confidences
    # )

    label_counts = count_coins(labels)

    total_value = compute_total_value(label_counts)

    annotated = draw_results(image, coins, labels)

    return {
        "annotated_image": annotated,
        "label_counts": label_counts,
        "total_value": total_value,
        "coins": coins,
        "labels": labels,
        "confidences": confidences,
        "top3_predictions": top3_predictions,
    }