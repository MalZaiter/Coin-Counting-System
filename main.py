from src.train import train
from src.predict import predict
from src.evaluate import evaluate

import cv2


def main():
    print("THIS IS MY NEW MAIN FILE")

    # =========================
    # TRAIN
    # =========================
    print("\nTRAINING MODEL...\n")

    train_results = train(
        labels_path="archive data/labels.csv",
        images_dir="archive data",
        model_type="svm"
    )

    print(train_results)

    # =========================
    # PREDICT
    # =========================
    print("\nRUNNING PREDICTION...\n")

    prediction = predict(
        image_path="tester.jpg",
        model_path="models/classifier.pkl",
        scaler_path="models/scaler.pkl"
    )

    print("Detected Coins:")
    print(prediction["label_counts"])

    print(f"\nTotal Value: €{prediction['total_value']:.2f}")

    cv2.imwrite(
        "outputs/result.jpg",
        prediction["annotated_image"]
    )

    print("\nAnnotated image saved.")

    # =========================
    # EVALUATE
    # =========================
    print("\nEVALUATING MODEL...\n")

    eval_results = evaluate(
        test_images_dir="archive data",
        labels_path="archive data/labels.csv",
        model_path="models/classifier.pkl",
        scaler_path="models/scaler.pkl"
    )

    print(f"\nFinal Accuracy: {eval_results['accuracy']:.4f}")


if __name__ == "__main__":
    main()