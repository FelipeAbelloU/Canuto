"""Extrae texto de PDFs con capa de texto seleccionable (no escaneados) a Markdown.

Extractor principal: **pymupdf4llm**, elegido tras el benchmark documentado en
``docs/comparacion_extraccion_md.md`` (reflowa párrafos, detecta tablas, rápido y
ligero). Devuelve Markdown directamente; la estructura legal se normaliza después en
``to_markdown.py``.

Fallbacks: pdfplumber y pypdf (texto plano) si pymupdf4llm falla o queda vacío.

Además detecta la **capa de texto corrupta** (fuentes con ToUnicode roto que
sustituyen letras por dígitos de forma sistemática, p.ej. "dicta11" en vez de
"dictan"). Esos PDFs no los arregla ningún extractor de texto: deben ir a la cola de
OCR (docling en la workstation).
"""
from __future__ import annotations

import re
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class ExtractedDocument:
    filename: str
    filepath: str
    text: str
    pages: int
    method: str
    corrupt: bool = False
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)

    def is_empty(self) -> bool:
        return self.char_count < 100


# --- Extractores (se prueban en orden hasta que uno devuelva texto) --------
def _try_pymupdf4llm(path: Path) -> str | None:
    # Extractor principal: entrega el texto ya en Markdown.
    try:
        import pymupdf4llm
        md = pymupdf4llm.to_markdown(str(path), show_progress=False)
        return md if md and md.strip() else None
    except Exception:
        return None


def _try_pdfplumber(path: Path) -> str | None:
    # Respaldo 1: texto plano página por página.
    try:
        import pdfplumber
        texts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
        return "\n\n".join(texts) if texts else None
    except Exception:
        return None


def _try_pypdf(path: Path) -> str | None:
    # Respaldo 2: último recurso si los anteriores fallan.
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
        return "\n\n".join(texts) if texts else None
    except Exception:
        return None


def _count_pages(path: Path) -> int:
    try:
        import pymupdf  # PyMuPDF
        with pymupdf.open(path) as doc:
            return doc.page_count
    except Exception:
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return len(pdf.pages)
        except Exception:
            return 0


# --- Detección de capa de texto corrupta ----------------------------------
# Trigrama letra-dígito-letra sin espacios: firma de fuente rota (ToUnicode).
_CORRUPT_TRIGRAM_RE = re.compile(r"[A-Za-zÁÉÍÓÚáéíóúÑñ][0-9]{1,2}[A-Za-zÁÉÍÓÚáéíóúÑñ]")


def corruption_score(text: str) -> int:
    """Puntúa la corrupción de fuente. Solo cuenta trigramas SISTEMÁTICOS (que se
    repiten ≥3 veces), para no confundir ruido disperso de tablas/códigos con una
    fuente rota real. Calibrado sobre el corpus: los limpios dan 0."""
    counts = Counter(m.lower() for m in _CORRUPT_TRIGRAM_RE.findall(text))
    return sum(v for v in counts.values() if v >= 3)


def looks_corrupt(text: str, threshold: int = 6) -> bool:
    """True si la capa de texto tiene corrupción sistemática de fuente."""
    return corruption_score(text) >= threshold


# --- API ------------------------------------------------------------------
def extract(pdf_path: str | Path) -> ExtractedDocument:
    """Extrae un PDF digital a Markdown. Marca ``corrupt`` si la fuente está rota."""
    path = Path(pdf_path)
    pages = _count_pages(path)

    # Probar los extractores en orden de calidad hasta obtener texto.
    text = _try_pymupdf4llm(path)
    method = "pymupdf4llm"

    if not text:
        text = _try_pdfplumber(path)
        method = "pdfplumber"
    if not text:
        text = _try_pypdf(path)
        method = "pypdf"

    text = text or ""
    return ExtractedDocument(
        filename=path.name,
        filepath=str(path.resolve()),
        text=text,
        pages=pages,
        method=method,
        corrupt=looks_corrupt(text),
    )


def is_likely_digital(pdf_path: str | Path, sample_pages: int = 3) -> bool:
    """Heurística: si se extraen ≥50 caracteres por página en promedio, es digital."""
    path = Path(pdf_path)
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            pages_to_check = pdf.pages[:sample_pages]
            if not pages_to_check:
                return False
            total_chars = sum(len(p.extract_text() or "") for p in pages_to_check)
            return (total_chars / len(pages_to_check)) >= 50
    except Exception:
        return False
