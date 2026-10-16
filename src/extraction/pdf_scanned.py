"""Extrae texto de PDFs escaneados (imagenes) usando OCR con docling.

docling hace el OCR y ademas reconoce tablas, y entrega el resultado ya en Markdown.
Necesita GPU para ser practico a gran escala y un ENTORNO VIRTUAL APARTE, porque sus
dependencias chocan con las del stack de entrenamiento/inferencia. Ver
``requirements-docling.txt``.

El import de docling es diferido: importar este modulo no falla en la laptop sin
docling; solo falla si se llama a ``extract()`` sin docling instalado.
"""
from __future__ import annotations

from pathlib import Path
from .pdf_digital import ExtractedDocument

# Idioma del reconocimiento de RapidOCR (el motor que usa docling en la workstation).
# Sin configurar, docling usa el modelo chino por defecto, y los sellos y firmas salen
# como caracteres chinos sueltos. "latin" cubre el español (tildes y eñe).
# Con None se usa la configuracion por defecto de docling (como en la extraccion de julio).
OCR_IDIOMA = "latin"

# El convertidor carga los modelos de OCR: se crea una sola vez y se reusa en cada PDF.
_converter = None


def _count_pages(path: Path) -> int:
    # Cuenta las paginas del PDF (solo para el reporte; no afecta al OCR).
    try:
        import pymupdf
        with pymupdf.open(str(path)) as doc:
            return doc.page_count
    except Exception:
        return 0


def extract(pdf_path: str | Path) -> ExtractedDocument:
    """Convierte un PDF escaneado a Markdown usando el OCR de docling."""
    path = Path(pdf_path)

    # Import diferido: docling solo se necesita en la maquina que corre el OCR.
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    except ImportError:
        raise ImportError(
            "docling no está instalado. Va en un venv APARTE (ver requirements-docling.txt).\n"
            "  Linux:    python3 -m venv venv-docling && source venv-docling/bin/activate\n"
            "  Windows:  python -m venv venv-docling && venv-docling\\Scripts\\activate\n"
            "  pip install -r requirements-docling.txt\n"
            "  (en Linux con GPU, más simple: bash install-docling.sh)"
        )

    global _converter
    if _converter is None:
        if OCR_IDIOMA is None:
            _converter = DocumentConverter()
        else:
            from rapidocr import LangRec
            # backend "torch": el que docling elige solo en la workstation (onnxruntime no
            # esta instalado). Pedir RapidOcrOptions a mano cambia el default a onnxruntime.
            ocr = RapidOcrOptions(
                backend="torch",
                rapidocr_params={"Rec.lang_type": LangRec(OCR_IDIOMA)},
            )
            opciones = PdfPipelineOptions(do_ocr=True, ocr_options=ocr)
            _converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opciones)}
            )

    # docling hace el OCR y devuelve el documento ya convertido a Markdown.
    result = _converter.convert(str(path))
    text = result.document.export_to_markdown()

    return ExtractedDocument(
        filename=path.name,
        text=text,
        pages=_count_pages(path),
        method="docling",
    )
