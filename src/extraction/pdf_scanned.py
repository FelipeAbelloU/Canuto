"""Extrae texto de PDFs escaneados (imágenes) usando OCR con docling.

docling hace el OCR y además reconoce tablas, y entrega el resultado ya en Markdown.
Necesita GPU para ser práctico a gran escala y un ENTORNO VIRTUAL APARTE, porque sus
dependencias chocan con las del stack de entrenamiento/inferencia. Ver
``requirements-docling.txt`` y ``docs/comparacion_extraccion_md.md``.

El import de docling es diferido: importar este módulo no falla en la laptop sin
docling; solo falla si se llama a ``extract()`` sin docling instalado.
"""
from __future__ import annotations

from pathlib import Path
from .pdf_digital import ExtractedDocument


def _count_pages(path: Path) -> int:
    # Cuenta las páginas del PDF (solo para el reporte; no afecta al OCR).
    try:
        import pymupdf
        with pymupdf.open(str(path)) as doc:
            return doc.page_count
    except Exception:
        return 0


def extract(pdf_path: str | Path) -> ExtractedDocument:
    """Convierte un PDF escaneado a Markdown usando el OCR de docling."""
    path = Path(pdf_path)

    # Import diferido: docling solo se necesita en la máquina que corre el OCR.
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

    # docling hace el OCR y devuelve el documento ya convertido a Markdown.
    converter = DocumentConverter()
    result = converter.convert(str(path))
    text = result.document.export_to_markdown()

    return ExtractedDocument(
        filename=path.name,
        filepath=str(path.resolve()),
        text=text,
        pages=_count_pages(path),
        method="docling",
    )
