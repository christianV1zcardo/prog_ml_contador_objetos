"""Contador de objetos en imágenes estáticas usando OpenCV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.preprocessing import apply_blur, to_gray
from src.segmentation import apply_threshold
from src.detection import find_objects
from src.visualization import draw_contours, save_image

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "output"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contador de objetos por contornos")
    parser.add_argument(
        "--image",
        default=str(IMAGES_DIR),
        help="Archivo o carpeta con imágenes a procesar (por defecto ./images)",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Archivo (una sola imagen) o carpeta donde guardar resultados",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=300.0,
        help="Área mínima del contorno a considerar (px²)",
    )
    parser.add_argument(
        "--min-circularity",
        type=float,
        default=0.2,
        help="Circularidad mínima (0-1) para descartar formas alargadas",
    )
    parser.add_argument(
        "--min-extent",
        type=float,
        default=0.4,
        help="Extensión mínima (área/bounding-box) para descartar ruido",
    )
    parser.add_argument(
        "--max-area",
        type=float,
        default=None,
        help="Área máxima permitida para cada contorno (px²). Si se omite se calcula automáticamente",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="No dibujar etiquetas numéricas ni cajas en el resultado",
    )
    return parser.parse_args(argv)


def load_image(path: Path) -> np.ndarray:
    """Load an image from disk, raising an explicit error when missing."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"No se encontró la imagen requerida en: {path}")
    return image


def _is_pattern(path: Path) -> bool:
    return any(char in str(path) for char in "*?[]")


def collect_input_paths(target: Path) -> List[Path]:
    """Return a sorted list of image paths based on a file, directory or patrón."""
    if _is_pattern(target):
        matches = sorted(Path(p) for p in Path().glob(str(target)))
        valid = [p for p in matches if p.is_file()]
        if not valid:
            raise FileNotFoundError(f"No hay imágenes que coincidan con el patrón: {target}")
        return valid

    if target.is_dir():
        candidates = sorted(p for p in target.iterdir() if p.is_file())
        filtered = [p for p in candidates if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not filtered:
            raise FileNotFoundError(
                f"La carpeta {target} no contiene imágenes con extensiones soportadas: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        return filtered

    if target.is_file():
        return [target]

    raise FileNotFoundError(f"No se encontró el archivo o carpeta: {target}")


def build_output_path(source: Path, output_arg: Path, multi: bool) -> Path:
    """Determine dónde guardar el resultado asociado a una imagen."""
    if not multi:
        if output_arg.is_dir() or output_arg.suffix == "":
            output_arg.mkdir(parents=True, exist_ok=True)
            return output_arg / f"{source.stem}_result.jpg"
        output_arg.parent.mkdir(parents=True, exist_ok=True)
        return output_arg

    # modo batch siempre escribe dentro de una carpeta
    output_dir = output_arg
    if output_dir.suffix:
        output_dir = output_dir.parent / output_dir.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{source.stem}_result.jpg"


def run_pipeline(
    input_path: Path,
    output_path: Path,
    *,
    min_area: float,
    min_circularity: float,
    min_extent: float,
    max_area: float | None,
    annotate: bool,
) -> int:
    """Execute the detection pipeline and return the contour count."""
    image = load_image(input_path)
    gray = to_gray(image)
    blurred = apply_blur(gray)
    thresh = apply_threshold(blurred)
    contours, count = find_objects(
        thresh,
        min_area=min_area,
        min_circularity=min_circularity,
        min_extent=min_extent,
        max_area=max_area,
    )
    visual = draw_contours(image, contours, annotate=annotate)
    save_image(output_path, visual)
    return count


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = Path(args.image)
    output_target = Path(args.output)

    try:
        sources = collect_input_paths(input_path)
        multi = len(sources) > 1

        total = 0
        for src in sources:
            destination = build_output_path(src, output_target, multi)
            count = run_pipeline(
                src,
                destination,
                min_area=args.min_area,
                min_circularity=args.min_circularity,
                min_extent=args.min_extent,
                max_area=args.max_area,
                annotate=not args.no_annotations,
            )
            total += count
            print(f"[{src.name}] Objetos detectados: {count}")

        if multi:
            print(f"Procesadas {len(sources)} imágenes. Total contornos: {total}")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - unexpected failures
        print(f"Error inesperado: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
