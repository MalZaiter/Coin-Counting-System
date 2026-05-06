import cv2
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detect import detect_coins
from src.features import extract_features, extract_features_batch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "data", "raw")

SAVE_OUTPUT = True
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

FEATURE_NAMES = [
    "radius", "area", "perimeter",
    "circularity", "aspect_ratio", "solidity",
    "hsv_h_mean", "hsv_s_mean", "hsv_v_mean",
    "h_hist_0", "h_hist_1", "h_hist_2", "h_hist_3", "h_hist_4", "h_hist_5",
    "s_hist_0", "s_hist_1", "s_hist_2", "s_hist_3", "s_hist_4",
    "v_hist_0", "v_hist_1", "v_hist_2", "v_hist_3", "v_hist_4",
    "edge_density", "grad_mean", "grad_std",
]

if SAVE_OUTPUT and not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


def print_features(coin_idx, fv):
    print(f"  Coin {coin_idx + 1}:")
    for name, val in zip(FEATURE_NAMES, fv):
        print(f"    {name:<20} {val:.4f}")


def main():
    image_files = sorted([f for f in os.listdir(DATASET_PATH) if f.endswith(".jpg")])

    if not image_files:
        print("No images found in dataset folder")
        return

    for filename in image_files:
        path = os.path.join(DATASET_PATH, filename)

        image = cv2.imread(path)
        if image is None:
            print(f"Failed to load {filename}")
            continue

        coins = detect_coins(image)

        if not coins:
            print(f"{filename} → no coins detected, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"{filename} → {len(coins)} coin(s) detected")
        print(f"{'='*50}")

        feature_matrix, valid_idx = extract_features_batch(image, coins)

        if feature_matrix.shape[0] == 0:
            print("Feature extraction failed for all coins")
            continue

        print(f"Feature vector length: {feature_matrix.shape[1]}")

        output = image.copy()

        for i, (vi, fv) in enumerate(zip(valid_idx, feature_matrix)):
            x, y, r = int(coins[vi][0]), int(coins[vi][1]), int(coins[vi][2])

            print_features(i, fv)

            circularity = fv[3]
            solidity    = fv[5]
            color = (0, 255, 0) if circularity > 0.8 and solidity > 0.9 else (0, 165, 255)

            cv2.circle(output, (x, y), r, color, 2)
            cv2.circle(output, (x, y), 3, (0, 0, 255), -1)

            label = f"#{i+1} r={int(r)} circ={circularity:.2f}"
            cv2.putText(output, label, (x - r, y - r - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        cv2.imshow("Feature Extraction", output)
        key = cv2.waitKey(500)

        if key == 27:
            break

        if SAVE_OUTPUT:
            save_path = os.path.join(OUTPUT_FOLDER, f"features_{filename}")
            cv2.imwrite(save_path, output)
            print(f"  Saved → {save_path}")

    cv2.destroyAllWindows()
    print("\n Feature extraction test complete.")


if __name__ == "__main__":
    main()
