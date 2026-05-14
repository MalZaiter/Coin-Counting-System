# Coin Counting System

A comprehensive computer vision pipeline for detecting, classifying, and counting coins in images using machine learning.

## Overview

This project implements an end-to-end system that:
- **Detects** coins in images using Hough circle detection with adaptive parameters
- **Extracts** feature vectors from detected coins (color, texture, and geometric properties)
- **Classifies** coins using machine learning (SVM or KNN classifiers)
- **Counts** and visualizes detected coins with detailed annotations
- **Evaluates** model performance with comprehensive metrics

## Features

### Detection Pipeline
- **Adaptive Hough Circle Detection**: Automatically adjusts parameters based on image characteristics
- **Multi-stage False Positive Filtering**: Four-stage filtering system to remove false detections
- **Advanced Deduplication**: 
  - Non-maximum suppression (NMS)
  - Arc-coverage merging
  - Cluster-wrapper removal
  - Nested-circle detection and removal

### Feature Extraction
- **Color Features**: HSV color space analysis and color histograms
- **Texture Features**: Edge density and gradient variation
- **Geometric Properties**: Size, position, and shape characteristics

### Classification
- **Multiple Classifiers**: Support for SVM and KNN algorithms
- **Feature Normalization**: Standardized feature scaling for robust predictions
- **Training/Test Split**: Group-based shuffle split for proper validation

## Project Structure

```
Coin-Counting-System/
├── main.py                 # End-to-end pipeline runner
├── data/
│   ├── raw/               # Raw input images
│   └── training/
│       ├── images/        # Training image dataset
│       ├── labels/        # YOLO format labels (*.txt files)
│       └── labels.csv     # CSV format labels
├── src/
│   ├── detect.py         # Hough circle detection & NMS
│   ├── features.py       # Feature extraction algorithms
│   ├── train.py          # Model training & evaluation
│   ├── predict.py        # Coin classification
│   ├── preprocess.py     # Image preprocessing utilities
│   ├── evaluate.py       # Model evaluation metrics
│   └── utils.py          # General utility functions
├── models/               # Saved trained models
│   ├── classifier.pkl    # Trained ML classifier
│   └── scaler.pkl        # Feature normalization scaler
└── outputs/
    ├── predictions/      # Classification predictions
    └── detections/       # Visualization with detections
```

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Coin-Counting-System
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

Run the complete pipeline (training + detection + visualization):
```bash
python main.py
```

### Individual Components

**Train a new model:**
```python
from src.train import train

results = train(
    labels_path="data/training/labels",
    images_dir="data/training",
    model_type="svm"  # or "knn"
)
print(f"Accuracy: {results['accuracy']:.4f}")
```

**Detect coins in an image:**
```python
import cv2
from src.detect import detect_coins, visualize_detections
from src.utils import load_image

image = load_image("path/to/image.jpg")
circles = detect_coins(image)
annotated = visualize_detections(image, circles)
cv2.imwrite("output.jpg", annotated)
```

**Classify detected coins:**
```python
from src.predict import predict

predictions = predict("path/to/image.jpg")
for coin_type, count in predictions.items():
    print(f"{coin_type}: {count}")
```

**Evaluate model performance:**
```python
from src.evaluate import evaluate

metrics = evaluate(
    labels_path="data/training/labels",
    images_dir="data/training"
)
```

## Data Format

### Label Format (YOLO .txt files)
Each `labels/*.txt` file corresponds to an image and contains one line per coin:
```
<class_id> <x_normalized> <y_normalized> <width_normalized> <height_normalized>
```
- Coordinates are normalized (0-1) relative to image dimensions
- `class_id`: 0=penny, 1=nickel, 2=dime, 3=quarter, etc.

### Label Format (CSV)
Alternative CSV format with columns:
```
image_path,x,y,radius,coin_type
```

## API Reference

### `src/detect.py`
- `detect_coins(image, method="hough", dp=1.2, param1=50, param2=30)` - Detect circles in image
- `non_maximum_suppression(circles, overlap_thresh=0.4)` - Remove overlapping detections
- `visualize_detections(image, circles, color=(0,255,0), thickness=2)` - Draw detected circles

### `src/features.py`
- `extract_features(image, circles, method="combined")` - Extract features from detected regions
- Supports: "color", "texture", "combined" methods

### `src/train.py`
- `train(labels_path, images_dir, model_type="svm")` - Train classifier and save model
- `load_training_data(labels_path, images_dir)` - Load labeled dataset

### `src/predict.py`
- `predict(image_path, return_features=False)` - Classify coins in image

### `src/preprocess.py`
- `enhance_contrast(image, method="clahe")` - Enhance image contrast
- `to_grayscale(image)` - Convert to grayscale
- `_scale_for_processing(image)` - Adaptive image scaling

### `src/utils.py`
- `load_image(path)` - Load image from file
- `normalize_features(features, scaler)` - Normalize feature vectors

## Configuration

Key parameters can be adjusted in `main.py` and individual modules:

- `MAX_PROCESS_DIM`: Maximum dimension for processing (default: 800px)
- `model_type`: Choice between "svm" or "knn" classifiers
- Hough detection parameters: `dp`, `param1`, `param2`, `minRadius`, `maxRadius`
- NMS overlap threshold: `overlap_thresh`

## Performance

- **Detection**: Handles images up to 4K resolution with automatic scaling
- **Classification**: Trained on 150+ labeled coin samples
- **Speed**: Real-time processing on standard hardware

## Output

### Generated Files
- `models/classifier.pkl` - Trained ML model
- `models/scaler.pkl` - Feature normalization parameters
- `outputs/predictions/` - Classification results
- `outputs/detections/` - Annotated detection visualizations

## Troubleshooting

### Detection Issues
- **No coins detected**: Adjust Hough parameters or improve image contrast
- **False positives**: Increase NMS overlap threshold or strengthen filters
- **Scale issues**: Check `MAX_PROCESS_DIM` setting for very large images

### Classification Issues
- **Poor accuracy**: Ensure training data is well-labeled and diverse
- **Memory errors**: Reduce image resolution or batch size

## Dependencies

Main libraries:
- `opencv-python` - Image processing and detection
- `scikit-learn` - Machine learning classifiers
- `numpy` - Numerical computations
- `joblib` - Model persistence

See `requirements.txt` for complete list.

## Future Improvements

- [ ] Deep learning integration (YOLO, Faster R-CNN)
- [ ] Real-time video processing
- [ ] Multi-currency support
- [ ] Mobile app deployment
- [ ] Web interface
- [ ] Augmented reality visualization


## Contributors

- Jana: Data collection & preprocessing
- Salma: Detection pipeline & post-processing
- Malak: Feature extraction
- Nour: Training & classification

## Contact

For questions or issues, please open an issue on the repository.
