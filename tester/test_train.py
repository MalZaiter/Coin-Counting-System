import cv2
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import load_from_csv, load_from_manifest, train, save_models, load_models, predict_single

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LABELS_CSV  = os.path.join(BASE_DIR, "data", "labels", "labels.csv")
MANIFEST    = os.path.join(BASE_DIR, "data", "labels", "manifest.json")
TRAIN_DIR   = os.path.join(BASE_DIR, "data", "train")
TEST_DIR    = os.path.join(BASE_DIR, "data", "test")

SAVE_OUTPUT = True
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

MODEL_TYPE = "svm"   # change to "knn" to test KNN instead

if SAVE_OUTPUT and not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


def load_data():
    """Load training data — prefer manifest, fall back to CSV."""
    if os.path.exists(MANIFEST):
        print(f"Loading data from manifest: {MANIFEST}")
        return load_from_manifest(MANIFEST)

    if os.path.exists(LABELS_CSV):
        print(f"Loading data from CSV: {LABELS_CSV}")
        return load_from_csv(TRAIN_DIR, LABELS_CSV)

    print("No labels found. Expected one of:")
    print(f"   {MANIFEST}")
    print(f"   {LABELS_CSV}")
    return [], []


def test_training(X, y_str):
    print(f"\n{'='*50}")
    print(f"Training {MODEL_TYPE.upper()} classifier")
    print(f"{'='*50}")

    clf, scaler, le = train(X, y_str, model_type=MODEL_TYPE, test_size=0.2)
    save_models(clf, scaler, le)

    print("\nModel trained and saved successfully.")
    print(f"   Classes: {list(le.classes_)}")
    return clf, scaler, le


def test_reload():
    print(f"\n{'='*50}")
    print("Reloading saved models from disk")
    print(f"{'='*50}")

    clf, scaler, le = load_models()
    print("Models loaded successfully.")
    print(f"   Classes: {list(le.classes_)}")
    return clf, scaler, le


def test_predictions(clf, scaler, le):
    """Run predict_single on every test image and display results."""
    if not os.path.exists(TEST_DIR):
        print(f"\nNo test folder found at {TEST_DIR}, skipping prediction test.")
        return

    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".jpg")])
    if not test_files:
        print("\nNo .jpg images found in data/test/, skipping prediction test.")
        return

    print(f"\n{'='*50}")
    print(f"Running predictions on {len(test_files)} test image(s)")
    print(f"{'='*50}")

    from src.detect import detect_coins

    for filename in test_files:
        path = os.path.join(TEST_DIR, filename)
        image = cv2.imread(path)
        if image is None:
            print(f"Failed to load {filename}")
            continue

        coins = detect_coins(image)
        if not coins:
            print(f"{filename} → no coins detected")
            continue

        print(f"\n{filename} → {len(coins)} coin(s) detected")

        output = image.copy()

        for i, circle in enumerate(coins):
            x, y, r = int(circle[0]), int(circle[1]), int(circle[2])

            label, confidence = predict_single(image, circle, clf, scaler, le)

            if label is None:
                print(f"  Coin {i+1}: feature extraction failed")
                continue

            conf_str = f"{confidence:.0%}" if confidence is not None else "n/a"
            print(f"  Coin {i+1}: {label:<12} confidence={conf_str}")

            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 3, (0, 0, 255), -1)

            tag = f"{label} ({conf_str})" if confidence is not None else label
            cv2.putText(output, tag, (x - r, y - r - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Prediction Test", output)
        key = cv2.waitKey(500)

        if key == 27:
            break

        if SAVE_OUTPUT:
            save_path = os.path.join(OUTPUT_FOLDER, f"pred_{filename}")
            cv2.imwrite(save_path, output)
            print(f"  Saved → {save_path}")

    cv2.destroyAllWindows()


def main():
    X, y_str = load_data()

    if not X:
        print("\nNo training data loaded — skipping training and prediction tests.")
        print(" Add labeled images to data/train/ and a labels file to data/labels/")
        return

    print(f"\nLoaded {len(X)} labeled samples")

    clf, scaler, le = test_training(X, y_str)

    clf, scaler, le = test_reload()

    test_predictions(clf, scaler, le)

    print("\nTraining test complete.")


if __name__ == "__main__":
    main()
