import cv2
from src.utils import load_image, show_image
from src.preprocess import to_grayscale, reduce_noise, equalize_normalize, edge_detection, preprocess

IMAGE_PATH = "data/raw/150.jpg"

def main():
  img = load_image(IMAGE_PATH)

  gray = to_grayscale(img)
  blur = reduce_noise(gray, method="gaussian")
  equalized = equalize_normalize(blur, method="normalize")
  edges = edge_detection(equalized)
  processed = preprocess(img)

  show_image("Original", img)
  show_image("Grayscale", gray)
  show_image("Blurred", blur)
  show_image("Equalized", equalized)
  show_image("Edges", edges)
  show_image("Preprocessed", processed)


if __name__ == "__main__":
    main()