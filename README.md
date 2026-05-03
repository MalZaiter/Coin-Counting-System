# Coin Counting System

A computer vision project that detects and classifies coins in images using **OpenCV's Circular Hough Transform** combined with size and color features.

---

## Overview

This system processes images containing coins and:

- Detects individual coins using the **Circular Hough Transform** (`cv2.HoughCircles`)
- Classifies each coin by analyzing its **radius (size)** and **average color (hue/brightness)**
- Counts the total number of coins and calculates the total monetary value
- Outputs an annotated image with each coin labeled

---

## Features

- Automatic coin detection via Circular Hough Transform
- Coin classification using radius and HSV color analysis
- Support for multiple coin denominations
- Annotated output image showing detected coins and their values
- Total coin count and monetary sum

---

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib (optional, for visualization)

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/coin-counting-system.git
   cd coin-counting-system
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate           # Windows
   ```
2. **Setup**

    ```bash
    pip install -r requirements.txt
    ```
---

## Project Structure

```
coin_counting_system/
│
├── data/
│   ├── raw/          # original images
│   ├── train/        # training images
│   ├── test/         # testing images
│   └── labels/       # labels (CSV or JSON)
│
├── models/
│   ├── coin_classifier.pkl
│   └── scaler.pkl
│
├── src/
│   ├── preprocess.py
│   ├── detect.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── utils.py
│
├── main.py
├── requirements.txt
└── README.md
```

## Usage

**Train:**
```bash
python main.py --mode train --images data/train --labels data/labels/train.csv
```

**Predict:**
```bash
python main.py --mode predict --image data/test/sample.jpg
```

**Evaluate:**
```bash
python main.py --mode evaluate --images data/test --labels data/labels/test.csv
```

## Pipeline

1. Image Input
2. Preprocessing (CLAHE, noise reduction, thresholding)
3. Circle Detection (Hough Transform)
4. False Object Rejection (contour analysis)
5. Feature Extraction (size, shape, color, texture)
6. Classification (KNN or SVM)
7. Output Visualization & Counting
