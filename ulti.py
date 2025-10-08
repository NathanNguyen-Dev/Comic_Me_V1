"""Utility helpers for the Comic Me Streamlit app."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from autocrop import Cropper

__all__ = ["adjust_gamma", "load_image_for_model", "load_frame_for_model"]


def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction to an image.

    Parameters
    ----------
    image:
        Input image represented as a NumPy array with pixel values in ``uint8``.
    gamma:
        Gamma value to apply. Values greater than 1 darken the image while values
        lower than 1 brighten it. The value is clamped to a small positive number
        to avoid division by zero.

    Returns
    -------
    np.ndarray
        The gamma-adjusted image.
    """
    safe_gamma = max(gamma, 1e-6)
    inv_gamma = 1.0 / safe_gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(
        "uint8"
    )
    return cv2.LUT(image, table)


def load_image_for_model(
    image: np.ndarray,
    *,
    crop_enabled: bool = False,
    zoom_percent: Optional[int] = None,
) -> tf.Tensor:
    """Prepare an uploaded image for inference.

    Parameters
    ----------
    image:
        The image uploaded by the user represented as an ``RGB`` array with
        values in the range ``[0, 255]``.
    crop_enabled:
        Whether the automatic face cropper should be applied prior to
        normalisation.
    zoom_percent:
        Percentage of the image to keep when cropping. Defaults to 50 when not
        provided.

    Returns
    -------
    tf.Tensor
        A 4-D tensor compatible with the CycleGAN model input.
    """
    working_image = image

    if crop_enabled:
        face_percent = zoom_percent or 50
        cropper = Cropper(face_percent=face_percent)
        cropped = cropper.crop(image)

        if cropped is not None:
            working_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        else:
            st.warning("Unable to detect a face to crop. Using the original image instead.")

    return _normalise_for_model(working_image)


def load_frame_for_model(image: np.ndarray) -> tf.Tensor:
    """Prepare a video frame for inference."""
    return _normalise_for_model(image)


def _normalise_for_model(image: np.ndarray) -> tf.Tensor:
    """Normalise an image array and reshape it for the model."""
    tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    tensor = (tensor / 255.0 * 2.0) - 1.0
    tensor = tf.image.resize(
        tensor,
        [256, 256],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )
    return tf.expand_dims(tensor, 0)
