"""Funciones de visualización para dibujar y guardar resultados."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def draw_contours(image: np.ndarray, contours: Sequence[np.ndarray], annotate: bool = True) -> np.ndarray:
    """Devuelve una copia de la imagen con contornos y anotaciones opcionales."""
    if image is None:
        raise ValueError("La imagen de entrada está vacía.")
    output = image.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    if annotate:
        for idx, contour in enumerate(contours, start=1):
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 1)

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.putText(
                output,
                str(idx),
                (cx - 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    return output


def save_image(path: Path | str, image: np.ndarray) -> None:
    """Guarda una imagen en disco y avisa si ocurre un error."""
    if image is None:
        raise ValueError("La imagen está vacía.")

    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(final_path), image):
        raise IOError(f"No se pudo guardar la imagen en {final_path}.")
