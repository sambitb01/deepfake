import streamlit as st
import tempfile
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# ==== IMAGE MODEL (unchanged) ====
from transformers import AutoImageProcessor, SiglipForImageClassification
import torch

IMAGE_MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"

@st.cache_resource
def load_image_model():
    processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL_NAME)
    model = SiglipForImageClassification.from_pretrained(IMAGE_MODEL_NAME)
    return processor, model

def predict_image(img, processor, model):
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    id2label = { "0": "fake", "1": "real" }
    return { id2label[str(i)]: probs[i] for i in range(len(probs)) }

# ==== VIDEO MODEL USING MESONET ====

@st.cache_resource
def load_video_model(path="mesonet_model.h5"):
    return load_model(path)

def predict_video(video_path, model, frame_count=30):
    cap = cv2.VideoCapture(video_path)
    frame_probs = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // frame_count, 1)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            face = cv2.resize(frame, (256, 256))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = np.expand_dims(face, axis=0) / 255.0
            prob = float(model.predict(face)[0][0])  # Assuming output is single score
            frame_probs.append(prob)
        idx += 1
    cap.release()
    if frame_probs:
        avg = sum(frame_probs) / len(frame_probs)
        return {"fake_score": round(avg, 3), "real_score": round(1 - avg, 3)}
    else:
        return {"fake_score": None, "real_score": None}

# ==== STREAMLIT APP ====

def main():
    st.title("Deepfake Detection: Image & Video (Public Model)")

    choice = st.radio("Select media type", ("Image", "Video"))
    if choice == "Image":
        uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)
            processor, model = load_image_model()
            scores = predict_image(img, processor, model)
            st.write("### Predictions")
            st.json(scores)

    else:
        uploaded = st.file_uploader("Upload a video", type=["mp4","avi","mov"])
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded.read())
            st.video(uploaded)

            st.write("Loading video detection model...")
            model = load_video_model()
            with st.spinner("Analyzing video frames..."):
                scores = predict_video(tfile.name, model)
            st.write("### Predictions (averaged)")
            st.json(scores)

if __name__ == "__main__":
    main()
