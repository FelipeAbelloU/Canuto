"""OCR de PDF escaneados con docling, que tambien reconoce tablas y entrega Markdown.
Va en un venv aparte (requirements-docling.txt) porque sus dependencias chocan con las del entrenamiento."""
from __future__ import annotations

from pathlib import Path
from .pdf_digital import ExtractedDocument

# El convertidor carga los modelos de OCR: se crea una sola vez y se reusa en cada PDF.
_converter = None


def _count_pages(path: Path) -> int:
    try:
        import pymupdf
        with pymupdf.open(str(path)) as doc:
            return doc.page_count
    except Exception:
        return 0


def extract(pdf_path: str | Path) -> ExtractedDocument:
    """Convierte un PDF escaneado a Markdown usando el OCR de docling."""
    path = Path(pdf_path)

    # se importa aqui para que el modulo cargue en el portatil sin docling
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise ImportError(
            "docling no está instalado. Va en un venv APARTE (ver requirements-docling.txt).\n"
            "  Linux:    python3 -m venv venv-docling && source venv-docling/bin/activate\n"
            "  Windows:  python -m venv venv-docling && venv-docling\\Scripts\\activate\n"
            "  pip install -r requirements-docling.txt\n"
            "  (en Linux con GPU, más simple: bash install-docling.sh)"
        )

    # Configuracion por defecto; los caracteres chinos de sellos y firmas se
    # quitan despues, en to_markdown.py.
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    result = _converter.convert(str(path))
    text = result.document.export_to_markdown()

    return ExtractedDocument(
        filename=path.name,
        text=text,
        pages=_count_pages(path),
        method="docling",
    )
