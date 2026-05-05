import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def reduce_noise(image: np.ndarray, method: str = "gaussian", ksize: int = 5) -> np.ndarray:
    """
    Noise reduction with Gaussian or Median blur.
    Use 3x3 or 5x5 kernels for coin images.
    """
    if ksize % 2 == 0:
        ksize += 1

    if method == "gaussian":
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif method == "median":
        return cv2.medianBlur(image, ksize)
    else:
        raise ValueError(f"Unsupported noise reduction method: {method}")


def enhance_contrast(image: np.ndarray, method: str = "clahe", color_space: str = "hsv") -> np.ndarray:
    """
    Contrast enhancement.

    Supported:
    - 'clahe'     : local contrast enhancement
    - 'equalize'  : global histogram equalization
    - 'normalize' : min-max normalization
    """
    if color_space == "hsv":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    v_clahe = clahe.apply(v)

    hsv = cv2.merge((h, s, v_clahe))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def threshold_image(image: np.ndarray, method: str = "otsu", invert: bool = False) -> np.ndarray:
    """
    Thresholding to separate coin regions from background.
    """
    if len(image.shape) == 3:
        image = to_grayscale(image)

    if method == "otsu":
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, thresh = cv2.threshold(image, 0, 255, thresh_type + cv2.THRESH_OTSU)
        return thresh
    elif method == "adaptive":
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            thresh_type,
            11,
            2
        )
    else:
        raise ValueError(f"Unsupported threshold method: {method}")


def morphological_operations(image: np.ndarray) -> np.ndarray:
    """
    Morphological cleanup:
    - opening removes small noise
    - closing fills small holes
    - erosion/dilation can separate/strengthen coin regions
    """
    kernel = np.ones((5, 5), np.uint8)

    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closed


def connected_component_filter(image: np.ndarray, min_area: int = 500) -> np.ndarray:
    """
    Remove small noise blobs using connected component analysis.
    Keeps only components with area >= min_area.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
    cleaned = np.zeros_like(image)

    for label in range(1, num_labels):  # skip background
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Canny edge detection.
    """
    return cv2.Canny(image, 80, 150)


def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Grayscale conversion
    2. Image resizing
    3. Noise reduction
    4. Contrast enhancement
    5. Thresholding
    6. Morphological operations
    7. Connected component filtering
    8. Edge detection
    """
    clahe_color = enhance_contrast(image, method="clahe")
    gray = to_grayscale(clahe_color)
    denoised = reduce_noise(gray, method="median", ksize=5)

    binary = threshold_image(denoised, method="otsu", invert=True)
    morphed = morphological_operations(binary)
    cleaned = connected_component_filter(morphed, min_area=700)

    edges = edge_detection(cleaned)
    return edges