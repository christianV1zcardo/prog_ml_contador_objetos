"""Utilidades de preprocesamiento de imágenes."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convierte una imagen BGR a escala de grises."""
    if image is None:
        raise ValueError("La imagen de entrada está vacía.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
    """Aplica desenfoque gaussiano para reducir ruido antes del umbral."""
    if image is None:
        raise ValueError("La imagen de entrada está vacía.")
    return cv2.GaussianBlur(image, kernel_size, 0)
