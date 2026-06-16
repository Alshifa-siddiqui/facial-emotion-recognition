# Facial Emotion Recognition

A convolutional neural network that classifies human facial expressions into
seven emotion categories, trained on the FER-2013 dataset. The repository
provides a complete pipeline: model training, held-out test evaluation with a
classification report and confusion matrix, and a real-time webcam demo built
on OpenCV.

## Overview

- **Task:** 7-class facial emotion classification (angry, disgust, fear, happy,
  neutral, sad, surprise).
- **Input:** 48x48 grayscale face crops, normalized to `[0, 1]`.
- **Model:** A compact CNN (three convolutional blocks followed by a dense
  classifier) trained from scratch.
- **Stack:** TensorFlow / Keras, OpenCV, scikit-learn, NumPy, Matplotlib.

## Model Architecture

| Stage | Layers |
|-------|--------|
| Block 1 | Conv2D(32, 3x3) -> BatchNorm -> MaxPool(2x2) |
| Block 2 | Conv2D(64, 3x3) -> BatchNorm -> MaxPool(2x2) |
| Block 3 | Conv2D(128, 3x3) -> BatchNorm -> MaxPool(2x2) |
| Head | Flatten -> Dense(256, ReLU) -> Dropout(0.5) -> Dense(7, Softmax) |

Training uses the Adam optimizer with categorical cross-entropy. Data
augmentation (rotation, width/height shift, zoom, horizontal flip) is applied to
the training set only. Training is regularized with early stopping, learning-rate
reduction on plateau, and best-checkpoint saving.

## Project Structure

```
facial-emotion-recognition/
|-- data/
|   |-- train/   (one subfolder per class)
|   |-- val/     (one subfolder per class)
|   |-- test/    (one subfolder per class)
|-- models/
|   |-- emotion_model.keras   (trained model, not tracked in git)
|   |-- emotion_model.h5       (legacy format, not tracked in git)
|   |-- confusion_matrix.png
|-- src/
|   |-- train.py              training pipeline
|   |-- test.py               test-set evaluation and confusion matrix
|   |-- realtime.py           webcam inference demo
|-- requirements.txt
|-- README.md
```

The `data/` directory and the trained model files are excluded from version
control via `.gitignore` because of their size. All scripts resolve paths
relative to the project root, so they behave identically whether invoked from
the repository root or from inside `src/`.

## Setup

Python 3.9+ is recommended. From the repository root:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

This project uses [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013),
arranged as one subfolder per class within each split:

```
data/train/<class>/*.png
data/val/<class>/*.png
data/test/<class>/*.png
```

Class names are inferred from the folder names in sorted order, which keeps the
training, evaluation, and inference label mappings consistent.

## Usage

### Train

```powershell
python src/train.py
```

The best model is checkpointed to `models/emotion_model.keras` (and a legacy
`.h5` copy) during training.

### Evaluate

```powershell
python src/test.py
```

Prints a per-class precision/recall/F1 report and saves the confusion matrix to
`models/confusion_matrix.png`.

### Real-time demo

```powershell
python src/realtime.py
```

Opens the default webcam, detects faces with a Haar cascade, and overlays the
predicted emotion and confidence. Press `q` to quit. If the default camera is
not index `0`, change `cv2.VideoCapture(0)` in `src/realtime.py`.

## Results

Evaluation on the held-out FER-2013 test set (3,586 images):

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|---------|
| Angry | 0.49 | 0.54 | 0.51 | 488 |
| Disgust | 0.62 | 0.09 | 0.16 | 55 |
| Fear | 0.42 | 0.34 | 0.38 | 528 |
| Happy | 0.82 | 0.85 | 0.83 | 879 |
| Neutral | 0.59 | 0.59 | 0.59 | 626 |
| Sad | 0.45 | 0.48 | 0.46 | 594 |
| Surprise | 0.71 | 0.74 | 0.72 | 416 |
| **Accuracy** | | | **0.60** | 3,586 |

Overall test accuracy is approximately 60%, which is in line with simple CNN
baselines on FER-2013.

## Known Limitations and Future Work

- **Class imbalance.** The `disgust` class is heavily under-represented (~1.5% of
  the training data) and is rarely recalled. Applying class weights or targeted
  oversampling is the most direct way to improve minority-class performance.
- **Architecture.** Accuracy can be improved with a deeper network, residual
  connections, or transfer learning from a pretrained backbone.
- **Inference preprocessing.** The webcam demo relies on a frontal-face Haar
  cascade; performance degrades under non-frontal poses, occlusion, or poor
  lighting.

## License

Released under the MIT License.
