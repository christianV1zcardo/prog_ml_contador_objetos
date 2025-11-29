"""Image segmentation helpers."""
from __future__ import annotations

import cv2
import numpy as np


def apply_threshold(
    gray_image: np.ndarray,
    thresh_value: int | None = None,
    invert: bool | None = None,
) -> np.ndarray:
    """Apply binary thresholding, usando Otsu por defecto para mayor robustez."""
    if gray_image is None:
        raise ValueError("Gray image is None.")

    if invert is None:
        # Objetos oscuros en fondo claro => invertir; caso contrario, modo normal.
        invert = gray_image.mean() > 127

    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if thresh_value is None:
        thresh_value = 0
        mode |= cv2.THRESH_OTSU

    _, thresh = cv2.threshold(gray_image, thresh_value, 255, mode)
    return thresh
