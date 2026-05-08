# Training Results

## Run Summary
- Script: `src/train.py`
- Classifier: `SVM`
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

## Metrics
- Accuracy: `0.7704`
- Precision: `0.7969`
- Recall: `0.7704`
- F1 Score: `0.7642`

## Saved Artifacts
- Model: `models/classifier.pkl`
- Scaler: `models/scaler.pkl`

## Notes
- The training script was updated to use the renamed archive folder: `archive data`
- The resolver also handles legacy label paths that still reference `archive/images`
- `complex_tests` remains unlabeled, so it is not part of training or scoring
