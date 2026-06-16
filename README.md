# Facial Emotion Recognition (FER-2013)

Detects human emotions from face images using a CNN on the **FER-2013** dataset. Trains a model, evaluates on a held-out test set, and runs a real-time webcam demo with OpenCV.

## 🔧 Tech

* TensorFlow / Keras
* OpenCV
* NumPy, scikit-learn, Matplotlib
* Dataset: FER-2013 (7 classes)

## 📁 Project structure

```
Facial Emotion Recognition/
│── data/
│   ├── train/  (angry, disgust, fear, happy, neutral, sad, surprise)
│   ├── val/    (same subfolders)
│   └── test/   (same subfolders)
│
│── models/
│   ├── emotion_model.keras
│   └── emotion_model.h5           # optional legacy save
│
│── notebooks/                     # optional experiments
│── src/
│   ├── train.py                   # training pipeline
│   ├── test.py                    # test-set evaluation + confusion matrix
│   └── realtime.py                # webcam demo
│
│── requirements.txt
│── README.md
```

## ⚙️ Setup

> Windows PowerShell (from the project root)

```powershell
pip install -r requirements.txt
```

If pip complains about permissions, add `--user`:

```powershell
pip install --user -r requirements.txt
```

## 🗂️ Dataset

1. Download FER-2013 (any version arranged in folders per class).
2. Ensure you have this split inside `data/`:

```
data/train/<class_folders>
data/val/<class_folders>
data/test/<class_folders>
```

If your zip wasn’t split, ask for the auto-split script (70/15/15).

## 🚀 Train

The scripts resolve paths relative to the project root, so they run the same
whether you call them from the repo root or from inside `src/`:

```powershell
# from repo root
python src/train.py
# or
cd src
python train.py
```

* Images are resized to `48×48`, grayscale, normalized.
* Model checkpoints to `models/emotion_model.keras` (and `.h5` legacy).

Expected baseline: \~55–60% val accuracy on a simple CNN. Use the “v2” training in the file (augmentation + callbacks) for a bump.

## ✅ Evaluate (Test Set)

```powershell
cd src
python test.py
```

Outputs:

* **Classification report** (precision/recall/F1 per class)
* **Confusion matrix** saved at: `models/confusion_matrix.png`

## 🎥 Real-Time Webcam Demo

```powershell
cd src
python realtime.py
```

* Press **q** to quit.
* If your default camera isn’t `0`, edit `VideoCapture(0)` to `1` or `2`.

## 🧪 Classes

The code auto-reads class names from `data/train/` folder names.
Typical FER-2013 classes:

```
angry, disgust, fear, happy, neutral, sad, surprise
```

## 🛠️ Troubleshooting

* **`No such file or directory: requirements.txt`**
  You’re in `src`. Run from project root:

  ```powershell
  pip install -r requirements.txt
  ```

  or

  ```powershell
  pip install -r ../requirements.txt
  ```

* **Model not found** when running `test.py` / `realtime.py`
  Make sure `train.py` finished and saved:

  ```
  models/emotion_model.keras
  ```

* **Black webcam window**
  Try `VideoCapture(1)` or `VideoCapture(2)`.

* **Slow training**
  Lower `epochs`, reduce `batch_size`, or train only the “v2” model with early stopping.

## 📦 Requirements

```
tensorflow>=2.10
opencv-python
numpy
matplotlib
scikit-learn
pillow
```

## 📝 How to Run (Quick Copy)

```powershell
# install
pip install -r requirements.txt

# train
python src/train.py

# evaluate
python src/test.py

# realtime demo
python src/realtime.py
```
