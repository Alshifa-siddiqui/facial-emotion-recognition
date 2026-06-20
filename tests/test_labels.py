"""Lightweight checks (stdlib only) for the persisted label mapping.

These run without TensorFlow/OpenCV so they execute quickly in CI.
"""
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_PATH = os.path.join(REPO_ROOT, "models", "labels.json")

EXPECTED = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def test_labels_json_exists():
    assert os.path.exists(LABELS_PATH), "models/labels.json is missing"


def test_labels_json_matches_fer2013_classes():
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    assert isinstance(labels, list)
    assert labels == EXPECTED, f"label order/content drifted: {labels}"


def test_labels_are_unique():
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    assert len(labels) == len(set(labels)) == 7
