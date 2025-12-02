"""Utilidades de detección basadas en el análisis de contornos."""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import cv2
import numpy as np

ContourList = Sequence[np.ndarray]


def _circularity(contour: np.ndarray) -> float:
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    area = cv2.contourArea(contour)
    return (4 * math.pi * area) / (perimeter ** 2)


def _extent(contour: np.ndarray) -> float:
    x, y, w, h = cv2.boundingRect(contour)
    box_area = max(w * h, 1)
    return cv2.contourArea(contour) / box_area


def find_objects(
    binary_image: np.ndarray,
    min_area: float = 300.0,
    min_circularity: float = 0.2,
    min_extent: float = 0.4,
    max_area: float | None = None,
) -> Tuple[List[np.ndarray], int]:
    """Encuentra contornos en una imagen binaria y los filtra por área y forma."""
    if binary_image is None:
        raise ValueError("La imagen binaria está vacía.")

    if max_area is None:
        height, width = binary_image.shape[:2]
        max_area = height * width * 0.1  # evita seleccionar bloques enormes (fondo)
    if max_area <= 0:
        max_area = float("inf")

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered: List[np.ndarray] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= min_area:
            continue
        if area >= max_area:
            continue
        if _circularity(cnt) < min_circularity:
            continue
        if _extent(cnt) < min_extent:
            continue
        filtered.append(cnt)

    return filtered, len(filtered)
