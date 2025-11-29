"""Fixtures compartidas para las pruebas del contador de objetos."""
from __future__ import annotations

import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_image() -> np.ndarray:
    """Imagen sintética con objetos oscuros sobre fondo claro."""
    image = np.full((200, 200, 3), 200, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (140, 160), (20, 20, 20), -1)
    cv2.circle(image, (150, 60), 25, (30, 30, 30), -1)
    return image
