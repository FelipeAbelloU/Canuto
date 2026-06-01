"""Extrae texto de PDFs escaneados (imágenes) usando OCR.
Soporta EasyOCR (recomendado) y Tesseract como alternativa.
Requiere pymupdf para convertir páginas PDF a imágenes.
"""
from __future__ import annotations

from pathlib import Path
from .pdf_digital import ExtractedDocument


def _pdf_to_images(path: Path) -> list:
    """Convierte cada página del PDF en una imagen PIL usando pymupdf."""
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError(
            "pymupdf no instalado. Ejecuta: pip install pymupdf Pillow"
        )
    import PIL.Image
    import io

    doc = fitz.open(str(path))
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = PIL.Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    return images


def _ocr_easyocr(images: list) -> str:
    try:
        import easyocr
        import numpy as np
    except ImportError:
        raise ImportError("easyocr no instalado. Ejecuta: pip install easyocr")

    reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
    pages_text = []
    for img in images:
        result = reader.readtext(np.array(img), detail=0, paragraph=True)
        pages_text.append(" ".join(result))
    return "\n\n".join(pages_text)


def _ocr_tesseract(images: list) -> str:
    try:
        import pytesseract
    except ImportError:
        raise ImportError(
            "pytesseract no instalado. Ejecuta: pip install pytesseract\n"
            "Además necesitas Tesseract instalado en el sistema: https://github.com/UB-Mannheim/tesseract/wiki"
        )
    pages_text = []
    for img in images:
        text = pytesseract.image_to_string(img, lang="spa")
        pages_text.append(text)
    return "\n\n".join(pages_text)


def extract(pdf_path: str | Path, engine: str = "easyocr") -> ExtractedDocument:
    """Extrae texto de un PDF escaneado usando OCR.

    Args:
        pdf_path: ruta al PDF escaneado
        engine: "easyocr" (por defecto) o "tesseract"
    """
    path = Path(pdf_path)
    images = _pdf_to_images(path)

    if engine == "tesseract":
        text = _ocr_tesseract(images)
        method = "tesseract"
    else:
        text = _ocr_easyocr(images)
        method = "easyocr"

    return ExtractedDocument(
        filename=path.name,
        filepath=str(path.resolve()),
        text=text,
        pages=len(images),
        method=method,
    )
