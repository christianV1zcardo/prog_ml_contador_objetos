"""Pruebas del pipeline de conteo de objetos."""
from __future__ import annotations

import cv2
import numpy as np

import main as main_module
from src.detection import find_objects
from src.preprocessing import apply_blur, to_gray
from src.segmentation import apply_threshold
from src.visualization import draw_contours, save_image


def test_preprocessing_pipeline(sample_image: np.ndarray) -> None:
    gray = to_gray(sample_image)
    assert gray.ndim == 2

    blurred = apply_blur(gray)
    assert blurred.shape == gray.shape
    assert not np.array_equal(gray, blurred)


def test_segmentation_threshold(sample_image: np.ndarray) -> None:
    gray = to_gray(sample_image)
    thresh = apply_threshold(gray)
    unique_vals = np.unique(thresh)
    assert set(unique_vals).issubset({0, 255})


def test_segmentation_auto_inverts() -> None:
    bright_bg = np.full((50, 50), 240, dtype=np.uint8)
    bright_bg[10:30, 10:30] = 20
    dark_bg = 255 - bright_bg

    thresh_dark_objs = apply_threshold(bright_bg, thresh_value=100)
    thresh_light_objs = apply_threshold(dark_bg, thresh_value=100)

    assert thresh_dark_objs[15, 15] == 255  # objeto oscuro detectado (invertido)
    assert thresh_light_objs[15, 15] == 255  # objeto claro detectado (no invertido)


def test_segmentation_manual_threshold_override() -> None:
    gradient = np.tile(np.linspace(0, 255, 50, dtype=np.uint8), (50, 1))
    manual = apply_threshold(gradient, thresh_value=128, invert=False)
    assert manual[:, :20].max() == 0
    assert manual[:, 30:].min() == 255


def test_detection_filters_by_area() -> None:
    binary = np.zeros((120, 160), dtype=np.uint8)
    cv2.rectangle(binary, (10, 10), (60, 80), 255, -1)  # área > 300
    cv2.rectangle(binary, (90, 20), (105, 35), 255, -1)  # área < 300

    contours, count = find_objects(binary, min_area=300, max_area=0)
    assert count == 1
    assert len(contours) == 1


def test_detection_filters_by_shape_metrics() -> None:
    binary = np.zeros((120, 160), dtype=np.uint8)
    cv2.rectangle(binary, (5, 5), (155, 25), 255, -1)  # rectángulo muy alargado

    contours, count = find_objects(
        binary,
        min_area=200,
        min_circularity=0.7,
        min_extent=0.5,
    )

    assert count == 0


def test_detection_auto_max_area_filters_large_regions() -> None:
    binary = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(binary, (0, 0), (199, 199), 255, -1)  # llena casi todo el fotograma

    contours, count = find_objects(binary, min_area=300)
    assert count == 0


def test_visualization_draw_and_save(tmp_path, sample_image: np.ndarray) -> None:
    contour = np.array([[[10, 10]], [[10, 60]], [[60, 60]], [[60, 10]]], dtype=np.int32)
    drawn = draw_contours(sample_image, [contour])
    assert drawn.shape == sample_image.shape
    assert np.any(drawn != sample_image)

    out_path = tmp_path / "result.jpg"
    save_image(out_path, drawn)
    assert out_path.exists()


def test_main_end_to_end(tmp_path, sample_image: np.ndarray, capsys) -> None:
    images_dir = tmp_path / "images"
    output_dir = tmp_path / "output"
    images_dir.mkdir()
    output_dir.mkdir()

    input_path = images_dir / "custom_input.jpg"
    result_path = output_dir / "custom_result.jpg"
    assert cv2.imwrite(str(input_path), sample_image)

    main_module.main([
        "--image",
        str(input_path),
        "--output",
        str(result_path),
    ])

    captured = capsys.readouterr()
    assert "Objetos detectados:" in captured.out
    assert result_path.exists()


def test_main_processes_directory(tmp_path, sample_image: np.ndarray, capsys) -> None:
    images_dir = tmp_path / "images"
    output_dir = tmp_path / "batch_output"
    images_dir.mkdir()

    for idx in range(2):
        path = images_dir / f"img_{idx}.jpg"
        assert cv2.imwrite(str(path), sample_image)

    main_module.main([
        "--image",
        str(images_dir),
        "--output",
        str(output_dir),
    ])

    captured = capsys.readouterr()
    assert "Procesadas 2 imágenes" in captured.out

    for idx in range(2):
        assert (output_dir / f"img_{idx}_result.jpg").exists()
