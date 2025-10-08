"""Streamlit entrypoint for the Comic Me application."""
from __future__ import annotations

import os
from typing import Optional

import cv2
import streamlit as st
import tensorflow as tf

from ulti import adjust_gamma, load_frame_for_model, load_image_for_model

MODEL_PATH = os.path.join("model", "ModelTrainOnKaggle.h5")


@st.cache_resource(show_spinner=True)
def load_model() -> tf.keras.Model:
    """Load the pre-trained CycleGAN model used for inference."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Expected the model weights to be available at 'model/ModelTrainOnKaggle.h5'."
        )
    return tf.keras.models.load_model(MODEL_PATH)


def render_intro() -> None:
    """Render the static introduction components."""
    st.set_page_config(layout="wide")
    st.image(os.path.join("Images", "Banner No2.png"), use_column_width=True)
    st.markdown(
        "<h1 style='text-align: center; color: white;'>Time to become a comic book character</h1>",
        unsafe_allow_html=True,
    )
    with st.expander("Configuration Options"):
        st.write("**AutoCrop** helps the model by finding and cropping the largest detectable face.")
        st.write("**Gamma Adjustment** can be used to lighten or darken the image.")


def render_image_mode(model: tf.keras.Model) -> None:
    """UI for converting an uploaded portrait to a comic styled image."""
    st.sidebar.header("Configuration")
    output_size = st.sidebar.selectbox("Output Size", options=[384, 512, 768], index=0)
    auto_crop = st.sidebar.checkbox("Auto Crop Image", value=True)
    zoom_percent: Optional[int] = None
    if auto_crop:
        zoom_percent = st.sidebar.slider(
            "Zoom adjust", min_value=50, max_value=100, value=50, step=5
        )
    gamma = st.sidebar.slider(
        "Gamma adjust", min_value=0.1, max_value=3.0, value=1.0, step=0.1
    )

    uploaded_file = st.file_uploader(
        "Upload your portrait here", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("Upload an image to start the transformation.")
        return

    col1, col2 = st.columns(2)
    image_bytes = uploaded_file.read()
    decoded_image = tf.image.decode_image(image_bytes, channels=3)
    decoded_image = decoded_image.numpy()
    adjusted_image = adjust_gamma(decoded_image, gamma=gamma)

    with col1:
        st.image(adjusted_image, caption="Original", use_column_width=True)

    input_tensor = load_image_for_model(
        adjusted_image,
        crop_enabled=auto_crop,
        zoom_percent=zoom_percent,
    )

    prediction = model(input_tensor, training=False)
    prediction = tf.squeeze(prediction, 0)
    prediction = (prediction * 0.5) + 0.5
    prediction = tf.image.resize(
        prediction,
        [output_size, output_size],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )

    with col2:
        st.image(prediction.numpy(), caption="Comic style", use_column_width=True)


def render_video_mode(model: tf.keras.Model) -> None:
    """UI for the optional video-based transformation mode."""
    run = st.checkbox("Run")
    frame_window = st.image([])
    camera = cv2.VideoCapture(0)
    gamma = st.slider(
        "Gamma adjust", min_value=0.1, max_value=3.0, value=1.0, step=0.1
    )

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        frame = adjust_gamma(frame, gamma=gamma)
        frame_tensor = load_frame_for_model(frame)
        prediction = model(frame_tensor, training=False)
        prediction = tf.squeeze(prediction, 0)
        prediction = (prediction * 0.5) + 0.5
        prediction = tf.image.resize(
            prediction,
            [384, 384],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )
        frame_window.image(prediction.numpy())

    camera.release()


def main() -> None:
    render_intro()
    model = load_model()

    menu = ["Image Based"]
    st.sidebar.header("Mode Selection")
    choice = st.sidebar.selectbox("How would you like to be turned?", menu)

    if choice == "Image Based":
        render_image_mode(model)
    elif choice == "Video Based":
        render_video_mode(model)


if __name__ == "__main__":
    main()
