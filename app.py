import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import pandas as pd
from keras.models import load_model
from keras.applications.xception import preprocess_input
from keras.preprocessing import image

# -----------------------------
# Load your trained model
# -----------------------------
MODEL_PATH = "deepfake_xception_demo.h5"   # change if you saved with .h5
model = load_model(MODEL_PATH)

# -----------------------------
# Helper functions
# -----------------------------
def extract_frames(video_path, max_frames=20):
    """Extract up to max_frames evenly spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames

def predict_on_frames(frames):
    """Run prediction on extracted frames."""
    preds = []
    for frame in frames:
        img = cv2.resize(frame, (150, 150))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        pred = model.predict(img, verbose=0)[0][0]
        preds.append(pred)
    return preds

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Deepfake Detection Demo")

uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save temp video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.video(uploaded_file)

    st.write("Extracting frames...")
    frames = extract_frames(tfile.name, max_frames=20)

    if len(frames) == 0:
        st.error("No frames extracted from video!")
    else:
        st.write(f"Extracted {len(frames)} frames")

        # Predict
        predictions = predict_on_frames(frames)
        avg_pred = np.mean(predictions)

        label = "FAKE" if avg_pred > 0.5 else "REAL"
        st.subheader(f"Prediction: **{label}** (Score: {avg_pred:.2f})")

        # Save frame-level predictions to CSV
        results_df = pd.DataFrame({
            "Frame": list(range(len(predictions))),
            "Prediction_Score": predictions
        })

        csv_path = "results.csv"
        results_df.to_csv(csv_path, index=False)

        # Add download button
        with open(csv_path, "rb") as f:
            st.download_button(
                label="Download Prediction Results (CSV)",
                data=f,
                file_name="deepfake_results.csv",
                mime="text/csv"
            )
