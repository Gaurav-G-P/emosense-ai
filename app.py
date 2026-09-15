import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="EmoSense AI - Facial Emotion Detector",
    page_icon="🎭",
    layout="wide"
)

st.title("🎭 EmoSense AI: Facial Emotion Recognition")
st.caption("Deep Learning-powered facial expression analysis")

st.sidebar.header("Input Source")
mode = st.sidebar.radio("Choose Input Mode:", ["Upload Image", "Webcam Capture"])

# Allow user to pick backend if lighting conditions vary
detector = st.sidebar.selectbox(
    "Face Detector Engine",
    ["yunet", "ssd", "opencv"],
    index=0,
    help="YuNet and SSD are far more accurate at cropping and aligning facial landmarks than basic OpenCV."
)

def analyze_emotion(image_np, selected_backend):
    from deepface import DeepFace

    # Convert RGB array from PIL/Streamlit to OpenCV BGR
    bgr_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Primary pass: aligned face detection
    try:
        analysis = DeepFace.analyze(
            img_path=bgr_img,
            actions=['emotion'],
            detector_backend=selected_backend,
            align=True,
            enforce_detection=True
        )
        return analysis[0] if isinstance(analysis, list) else analysis
    except Exception:
        # Secondary fallback: softer detection threshold with SSD
        try:
            analysis = DeepFace.analyze(
                img_path=bgr_img,
                actions=['emotion'],
                detector_backend='ssd',
                align=True,
                enforce_detection=False
            )
            return analysis[0] if isinstance(analysis, list) else analysis
        except Exception as e:
            st.error(f"Inference error: {e}")
            return None

image = None

if mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, PNG, JPEG)...", 
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        pil_image = Image.open(uploaded_file)
        image = np.array(pil_image)

elif mode == "Webcam Capture":
    camera_input = st.camera_input("Capture a live face frame:")
    if camera_input:
        pil_image = Image.open(camera_input)
        image = np.array(pil_image)

if image is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Deep Learning Prediction")
        with st.spinner("Detecting face & analyzing micro-expressions..."):
            result = analyze_emotion(image, detector)

        if result:
            dominant = result.get("dominant_emotion", "Unknown").capitalize()
            emotions = result.get("emotion", {})

            # Dominant metric display
            st.metric(label="Dominant Emotion", value=dominant)

            # Format confidence data
            df = pd.DataFrame(list(emotions.items()), columns=["Emotion", "Confidence (%)"])
            df["Emotion"] = df["Emotion"].str.capitalize()
            df["Confidence (%)"] = df["Confidence (%)"].round(2)
            df = df.sort_values(by="Confidence (%)", ascending=True)

            # Distribution chart
            fig = px.bar(
                df,
                x="Confidence (%)",
                y="Emotion",
                orientation="h",
                color="Confidence (%)",
                color_continuous_scale="Tealgrn",
                title="Model Confidence Breakdown"
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No face detected. Please ensure your face is well-lit and facing the camera directly.")