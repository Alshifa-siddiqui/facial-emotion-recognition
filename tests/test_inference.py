"""End-to-end test of the inference pipeline in app.py.

A tiny randomly-initialised CNN stands in for the trained model so the test
exercises the real face-detection -> preprocess -> predict -> annotate path
without requiring the (gitignored) trained weights. Skips cleanly if the heavy
optional dependencies are unavailable.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
tf = pytest.importorskip("tensorflow")

import app  # noqa: E402  (imported after dependency checks)


@pytest.fixture(scope="module")
def dummy_model():
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input

    model = Sequential([
        Input(shape=(48, 48, 1)),
        Conv2D(4, (3, 3), activation="relu"),
        Flatten(),
        Dense(len(app.FALLBACK_LABELS), activation="softmax"),
    ])
    return model


def test_fallback_labels_count():
    assert len(app.FALLBACK_LABELS) == 7


def test_load_labels_returns_seven():
    labels = app.load_labels()
    assert len(labels) == 7


def test_predict_faces_no_face_falls_back_to_whole_image(dummy_model):
    # Random noise: the cascade finds no face, so the code classifies the
    # whole frame. Expect exactly one result with a valid label/confidence.
    img = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    annotated, results = app.predict_faces(dummy_model, app.FALLBACK_LABELS, img)
    assert annotated.shape == img.shape
    assert len(results) == 1
    label, conf, probs = results[0]
    assert label in app.FALLBACK_LABELS
    assert 0.0 <= conf <= 1.0
    assert len(probs) == 7
