import cv2
from src.utils import load_image, show_image
from src.preprocess import to_grayscale, reduce_noise, equalize_normalize, threshold_image, morphological_operations, edge_detection, preprocess

IMAGE_PATH = "data/raw/150.jpg"

def main():
  img = load_image(IMAGE_PATH)

  gray = to_grayscale(img)
  blur = reduce_noise(gray, method="gaussian")
  normalize = equalize_normalize(blur, method="equalize")
  cleaned = morphological_operations(normalize)
  edges = edge_detection(cleaned)
  processed = preprocess(img)

  show_image("Original", img)
  show_image("Grayscale", gray)
  show_image("Blurred", blur)
  show_image("Normalized", normalize)
  show_image("Cleaned", cleaned)
  show_image("Edges", edges)
  show_image("Preprocessed", processed)


if __name__ == "__main__":
    main()