# Facial Emotion Recognition

A convolutional neural network that classifies human facial expressions into
seven emotion categories, trained on the FER-2013 dataset. The repository
provides a complete, reproducible pipeline: model training, held-out test
evaluation (classification report and confusion matrix), and a real-time webcam
inference demo built on OpenCV.

## Overview

This project trains a compact CNN from scratch on 48x48 grayscale face crops and
exposes three entry points:

- `src/train.py` builds and trains the model with augmentation and callbacks.
- `src/test.py` evaluates a saved model on the held-out test split.
- `src/realtime.py` runs live inference from a webcam using Haar-cascade face
  detection.

Measured baseline performance on the FER-2013 test split is approximately
**60% accuracy**, consistent with simple CNN baselines on this dataset.

## Features

- End-to-end training pipeline with on-the-fly data augmentation.
- Regularized training (early stopping, LR reduction on plateau, best-checkpoint
  saving).
- Reproducible evaluation with per-class precision/recall/F1 and a saved
  confusion matrix.
- Real-time webcam inference with face detection and confidence overlay.
- Browser-based image-upload demo (`app.py`, Streamlit) for hosted deployment
  without webcam hardware.
- Path-independent scripts (run from the repository root or from `src/`).
- Model persisted in both the native Keras format and legacy HDF5.
- Class labels persisted to `models/labels.json` so inference is decoupled from
  the (gitignored) `data/` directory.
- Headless test suite (`tests/`) and GitHub Actions CI.

## Technology Stack

| Area | Tools |
|------|-------|
| Modeling | TensorFlow / Keras |
| Computer vision | OpenCV (Haar cascade, webcam I/O) |
| Data / numerics | NumPy |
| Evaluation | scikit-learn, Matplotlib |
| Image I/O | Pillow |
| Language | Python 3.9–3.12 |

## Folder Structure

```
facial-emotion-recognition/
|-- data/                         (not tracked; user-provided dataset)
|   |-- train/<class>/*.png
|   |-- val/<class>/*.png
|   |-- test/<class>/*.png
|-- models/
|   |-- emotion_model.keras       (produced by training; not tracked)
|   |-- emotion_model.h5          (legacy format; not tracked)
|   |-- labels.json               (class label order for inference)
|   |-- confusion_matrix.png      (produced by evaluation)
|-- src/
|   |-- train.py
|   |-- test.py
|   |-- realtime.py
|-- tests/                        (headless pytest suite)
|-- .github/workflows/ci.yml      (CI: install + tests)
|-- app.py                        (Streamlit image-upload web demo)
|-- requirements.txt              (dev/training deps, unpinned)
|-- requirements-deploy.txt       (pinned runtime deps for the web app)
|-- LICENSE
|-- README.md
```

The `data/` directory and trained model weights are excluded from version
control because of their size. All scripts resolve paths relative to the project
root, so they behave identically regardless of the working directory.

## Installation

Requires Python 3.9 or newer (verified on 3.12).

```powershell
# Windows PowerShell, from the repository root
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset Requirements

This project uses [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013).
Download it and arrange one subfolder per class within each split:

```
data/train/<class>/*.png
data/val/<class>/*.png
data/test/<class>/*.png
```

Expected classes: `Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise`.
Class names are inferred from folder names in sorted order, which keeps the
training, evaluation, and inference label mappings consistent. The published
FER-2013 distribution typically ships only `train` and `test`; create a `val`
split by holding out a portion of `train` (for example 10–15%).

## Usage

### Training

```bash
python src/train.py
```

Trains for up to 40 epochs with early stopping. The best model is checkpointed to
`models/emotion_model.keras`, with a legacy `models/emotion_model.h5` copy saved
on completion.

### Evaluation

```bash
python src/test.py
```

Prints a per-class precision/recall/F1 report and writes the confusion matrix to
`models/confusion_matrix.png`.

### Real-Time Inference

```bash
python src/realtime.py
```

Opens the default webcam, detects faces, and overlays the predicted emotion and
confidence. Press `q` to quit. If the default camera is not index `0`, edit
`cv2.VideoCapture(0)` in `src/realtime.py`.

### Web App (image upload, no webcam)

```bash
pip install -r requirements-deploy.txt
streamlit run app.py
```

`app.py` serves a browser UI: upload a JPG/PNG, and the app detects faces,
overlays the predicted emotion + confidence, and shows a per-class probability
bar chart. If no face is detected it classifies the whole image. This is the
deployment-friendly entry point (no camera hardware required) and the reference
implementation referenced under [Deployment](#deployment).

## Testing

A headless `pytest` suite lives in `tests/`. It needs no webcam and no dataset:
label-mapping checks run on stdlib only, and the inference-pipeline test uses a
tiny in-memory dummy model so it does not require the (gitignored) trained
weights.

```bash
pip install pytest
pytest tests/ -v
```

CI (`.github/workflows/ci.yml`) runs the same suite on every push and pull
request.

## Deployment

The webcam demo requires direct camera hardware access and is intended to run
locally; it is not suitable for stateless cloud hosting as-is. Two supported
deployment paths:

**1. Containerized batch/evaluation image (recommended for reproducibility)**

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "src/test.py"]
```

```bash
docker build -t fer .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" fer
```

**2. Web service for image-upload inference (recommended for a public demo)**

This repository already ships that app: [`app.py`](app.py) is a Streamlit
service that accepts an uploaded image and returns the predicted emotion,
avoiding the webcam-hardware constraint. Deploy it to Streamlit Community Cloud,
Hugging Face Spaces, or any container platform (Render, Fly.io, Cloud Run) using
the pinned `requirements-deploy.txt`. The trained weights
(`models/emotion_model.keras`) are gitignored, so include them in the deploy
artifact (commit to the deployment branch, bake into the image, or pull from
object storage at startup).

```bash
pip install -r requirements-deploy.txt
streamlit run app.py
```

## Troubleshooting

| Symptom | Cause | Resolution |
|---------|-------|------------|
| `Found 0 images belonging to 0 classes` | `data/` not populated | Download FER-2013 and arrange the per-class folders as above. |
| `No saved model found` | Model not yet trained | Run `python src/train.py` first. |
| `Could not open camera 0` | Wrong camera index or no camera | Change `VideoCapture(0)` to `1`/`2`, or run on a machine with a webcam. |
| `libGL.so.1: cannot open shared object file` | Missing OpenCV system libs in container | Install `libgl1` and `libglib2.0-0` (see Dockerfile). |
| Very slow training | CPU-only execution | Use a GPU build of TensorFlow, reduce `epochs`/`batch_size`. |

## Known Limitations

- **Class imbalance.** `Disgust` is heavily under-represented (~1.5% of training
  data) and is rarely recalled (recall ≈ 0.09 at baseline).
- **Modest accuracy.** ~60% test accuracy reflects a small from-scratch CNN
  (measured: weighted F1 0.60, macro F1 0.52 on the FER-2013 test split).
- **Unpinned training dependencies.** `requirements.txt` is unpinned; the
  deployment runtime is pinned in `requirements-deploy.txt`.
- **Frontal-face assumption.** The Haar cascade degrades under pose, occlusion,
  and poor lighting.

## Future Improvements

- Apply class weights or oversampling to address imbalance (esp. `Disgust`).
- Pin training dependency versions (`requirements.txt`) or add a lockfile.
- Add a config layer (CLI args or a config file) to remove hard-coded
  hyperparameters.
- Expand the test suite (e.g. evaluation-metric regression tests gated on the
  presence of weights + dataset).
- Explore transfer learning or a deeper/residual architecture.

## License

Released under the [MIT License](LICENSE).
