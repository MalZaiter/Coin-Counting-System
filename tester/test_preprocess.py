

import cv2
from src.utils import load_image, show_image
from src.preprocess import to_grayscale, reduce_noise, enhance_contrast, threshold_image, morphological_operations, connected_component_filter, edge_detection, preprocess

IMAGE_PATH = "data/raw/023.jpg"

def main():
  img = load_image(IMAGE_PATH)

  contrast = enhance_contrast(img, method="clahe")
  gray = to_grayscale(contrast)
  denoised = reduce_noise(gray, method="median", ksize=5)
  binary = threshold_image(denoised, method="otsu", invert=False)
  morphed = morphological_operations(binary)
  cleaned = connected_component_filter(morphed, min_area=700)
  edges = edge_detection(cleaned)
  processed = preprocess(img)

  show_image("Original", img)
  show_image("CLAHE", contrast)
  show_image("Grayscale", gray)
  show_image("Blurred", denoised)
  show_image("Binary", binary)
  show_image("Morphed", morphed)
  show_image("Cleaned", cleaned)
  show_image("Edges", edges)
  show_image("Preprocessed", processed)


if __name__ == "__main__":
    main()