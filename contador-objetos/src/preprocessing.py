"""Image preprocessing utilities."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""
    if image is None:
        raise ValueError("Input image is None.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """Apply Gaussian blur to reduce noise before thresholding."""
    if image is None:
        raise ValueError("Input image is None.")
    return cv2.GaussianBlur(image, kernel_size, 0)
