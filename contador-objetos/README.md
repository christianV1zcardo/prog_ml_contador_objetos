# Contador de Objetos con OpenCV

Proyecto base en Python para detectar y contar objetos en una imagen fija utilizando OpenCV. El pipeline aplica preprocesamiento, segmentación, detección de contornos y visualización del resultado final.

## Requisitos

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python main.py
```

Por defecto procesará **todas** las imágenes soportadas (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`) que encuentre dentro de `./images/` y guardará cada resultado en `./output/<nombre>_result.jpg`.

También puedes pasar rutas personalizadas:

```bash
python main.py --image ruta/a/mi_imagen.jpg --output ruta/de/salida.jpg   # archivo único
python main.py --image ruta/a/una_carpeta --output resultados/             # carpeta completa
python main.py --image "images/*.jpeg" --output resultados/               # patrón tipo glob
```

### Parámetros útiles

- `--min-area`: descarta contornos muy pequeños (por defecto 300 px²).
- `--max-area`: limita el tamaño máximo (por defecto se calcula automáticamente). Usa `--max-area 0` para desactivar este filtro.
- `--min-circularity`: filtra objetos no circulares (0-1, por defecto 0.2).
- `--min-extent`: controla cuánta área ocupa el contorno respecto a su bounding box (por defecto 0.4).
- `--no-annotations`: si no quieres cajas ni números sobre la imagen resultante.

La imagen de salida dibuja el contorno en verde, la caja delimitadora en azul y una etiqueta roja numerada para que sepas qué contornos se están contando.

## Pruebas

```bash
pytest
```

Las pruebas unitarias validan cada etapa (preprocesamiento, segmentación, detección, visualización y ejecución completa del `main`).

## Flujo de procesamiento

1. **Preprocesamiento (`src/preprocessing.py`)**: conversión a escala de grises y desenfoque gaussiano para reducir ruido.
2. **Segmentación (`src/segmentation.py`)**: umbralización binaria para separar objetos del fondo.
3. **Detección (`src/detection.py`)**: búsqueda de contornos y filtrado por área mínima (> 300 px).
4. **Visualización (`src/visualization.py`)**: dibujo de contornos sobre la imagen original y guardado del resultado.

## Estructura del proyecto

```text
contador-objetos/
├── images/
│   └── input.jpg
├── main.py
├── output/
├── README.md
├── requirements.txt
└── src/
    ├── detection.py
    ├── preprocessing.py
    ├── segmentation.py
    └── visualization.py
```

## Ejemplo de salida

```text
Objetos detectados: 3
```

La imagen resultante se encontrará en `./output/result.jpg` mostrando los contornos dibujados en verde.
