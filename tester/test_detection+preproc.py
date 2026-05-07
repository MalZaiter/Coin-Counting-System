import cv2
import os
from src.detect import detect_coins

# Path to your dataset
import os


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "data", "complex_tests")

# Optional: save results instead of just showing
SAVE_OUTPUT = True
OUTPUT_FOLDER = os.path.join(BASE_DIR, "complext_tests_outputs")

# Set to a filename (e.g., "128.jpg") to process only that image, or None to process all
SPECIFIC_IMAGE = None  # e.g., "128.jpg" or None

if SAVE_OUTPUT and not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


def main():
    if SPECIFIC_IMAGE:
        image_files = [SPECIFIC_IMAGE] if os.path.exists(os.path.join(DATASET_PATH, SPECIFIC_IMAGE)) else []
        if not image_files:
            print(f"❌ Image {SPECIFIC_IMAGE} not found in dataset folder")
            return
    else:
        image_files = sorted([f for f in os.listdir(DATASET_PATH) if f.endswith(".jpg")])

    if not image_files:
        print("❌ No images found in dataset folder")
        return

    for filename in image_files:
        path = os.path.join(DATASET_PATH, filename)

        # Load image
        image = cv2.imread(path)
        if image is None:
            print(f"❌ Failed to load {filename}")
            continue

        # Run detection
        coins = detect_coins(image)

        print(f"{filename} → {len(coins)} coins detected")

        # Draw results
        output = image.copy()

        for (x, y, r) in coins:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)   # coin boundary
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)   # center point

        # Show image
        cv2.imshow("Detection", output)
        key = cv2.waitKey(500)  # show for 0.5 sec

        # Press ESC to stop early
        if key == 27:
            break

        # Save output image
        if SAVE_OUTPUT:
            save_path = os.path.join(OUTPUT_FOLDER, filename)
            cv2.imwrite(save_path, output)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()