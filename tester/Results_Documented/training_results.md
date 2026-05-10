# Training Results

## Run Summary
- Script: `src/train.py`
- Labels file: `archive data/labels.csv`
- Images folder: `archive data/images`
- Split: 80% train / 20% test

## Dataset
- Total labeled samples: `672`
- Classes:
  - `10cent`
  - `1cent`
  - `1euro`
  - `20cent`
  - `2cent`
  - `2euro`
  - `50cent`
  - `5cent`

## Train/Test Split
- Training samples: `537`
- Test samples: `135`

## Model Comparison

| Metric | KNN | SVM | Winner |
|--------|-----|-----|--------|
| **Accuracy** | 0.7259 | 0.7704 | SVM ✓ |
| **Precision** | 0.7324 | 0.7969 | SVM ✓ |
| **Recall** | 0.7259 | 0.7704 | SVM ✓ |
| **F1 Score** | 0.7245 | 0.7642 | SVM ✓ |

## Analysis

- **SVM outperforms KNN** across all metrics (4.5% improvement in accuracy)
- SVM shows better generalization with higher precision and recall
- SVM is the preferred model for this coin classification task
- Both models are acceptable but SVM provides superior performance

## Saved Artifacts
- Model: `models/classifier.pkl` (SVM - recommended)
- Scaler: `models/scaler.pkl`

## Notes
- The training script was updated to use the renamed archive folder: `archive data`
- The resolver also handles legacy label paths that still reference `archive/images`
- `complex_tests` remains unlabeled, so it is not part of training or scoring
- Both KNN and SVM models have been trained and evaluated
- The SVM model is saved as the primary classifier due to superior performance
