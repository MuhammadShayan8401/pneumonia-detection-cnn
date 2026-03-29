# streamlit_app.py

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------------
# Load Model
# -------------------------
model = load_model('../models/cnn_model.h5')

# -------------------------
# Preprocess Function
# -------------------------
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------
# Streamlit UI Config
# -------------------------
st.set_page_config(page_title="Disease Detection", layout="centered")

# -------------------------
# Sidebar (NEW)
# -------------------------
st.sidebar.title("About")
st.sidebar.write("AI-based Pneumonia Detection using CNN")
st.sidebar.write("Upload a chest X-ray image to detect Pneumonia using deep learning.")
st.sidebar.write("Model Accuracy: ~90%")

# -------------------------
# Main UI
# -------------------------
st.title("🧠 AI Disease Detection from X-ray")
st.write("Upload a chest X-ray image to detect Pneumonia")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Show uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    processed_image = preprocess_image(image)

    # Prediction
    prediction = model.predict(processed_image)[0][0]

    # Result Logic
    if prediction > 0.5:
        result = "🦠 Pneumonia Detected"
        confidence = float(prediction)
        is_pneumonia = True
    else:
        result = "✅ Normal"
        confidence = float(1 - prediction)
        is_pneumonia = False

    # -------------------------
    # Display Results
    # -------------------------
    st.subheader("Prediction Result")

    # Better UI (NEW)
    if is_pneumonia:
        st.error(result)
    else:
        st.success(result)

    # Confidence
    st.write(f"Confidence: {confidence * 100:.2f}%")

    # Progress Bar (NEW)
    st.progress(confidence)

    # Extra Info (NEW)
    st.write("Model Accuracy: ~90%")