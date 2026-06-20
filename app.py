"""Streamlit web app for facial emotion recognition.

Accepts an uploaded image, detects faces with a Haar cascade, and predicts the
emotion for each face using the trained CNN. Designed for hosted deployment
(Hugging Face Spaces / Render / any container platform) where webcam access is
unavailable.
"""
import os
import json
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = [
    os.path.join(BASE_DIR, "models", "emotion_model.keras"),
    os.path.join(BASE_DIR, "models", "emotion_model.h5"),
]
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.json")
FALLBACK_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


@st.cache_resource
def load_emotion_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return load_model(path), path
    return None, None


def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            return json.load(f)
    return FALLBACK_LABELS


def predict_faces(model, labels, image_bgr):
    """Return (annotated_image, results) where results is a list of
    (label, confidence, all_probs)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # If no face is detected, fall back to classifying the whole image.
    if len(faces) == 0:
        h, w = gray.shape
        faces = [(0, 0, w, h)]

    results = []
    annotated = image_bgr.copy()
    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48)).astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))
        probs = model.predict(roi, verbose=0)[0]
        idx = int(np.argmax(probs))
        label, conf = labels[idx], float(probs[idx])
        results.append((label, conf, probs))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, f"{label} {conf * 100:.0f}%", (x, max(y - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return annotated, results


def main():
    st.set_page_config(page_title="Facial Emotion Recognition", page_icon="🙂")
    st.title("Facial Emotion Recognition")
    st.write(
        "Upload a photo of a face and the model predicts one of seven emotions: "
        "angry, disgust, fear, happy, neutral, sad, surprise."
    )

    model, model_path = load_emotion_model()
    if model is None:
        st.error("No trained model found in models/. Train the model first "
                 "(python src/train.py) and ensure the weights are deployed.")
        st.stop()
    labels = load_labels()
    st.caption(f"Loaded model: {os.path.basename(model_path)} | classes: {', '.join(labels)}")

    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload a JPG or PNG to get started.")
        return

    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("Could not read that image. Try a different file.")
        return

    annotated, results = predict_faces(model, labels, image_bgr)
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
             caption="Detected faces", use_container_width=True)

    for i, (label, conf, probs) in enumerate(results, 1):
        st.subheader(f"Face {i}: {label} ({conf * 100:.1f}%)")
        st.bar_chart({lbl: float(p) for lbl, p in zip(labels, probs)})


if __name__ == "__main__":
    main()
