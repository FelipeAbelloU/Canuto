"""Extrae PDFs digitales (con capa de texto) a Markdown.

Usa pymupdf4llm (elegido en docs/comparacion_extraccion_md.md); si falla,
cae a pdfplumber. También marca la capa de texto corrupta para desviarla a OCR.
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


def _try_pymupdf4llm(path: Path) -> str | None:
    try:
        import pymupdf4llm
        md = pymupdf4llm.to_markdown(str(path), show_progress=False)
        return md if md and md.strip() else None
    except Exception:
        return None


def _try_pdfplumber(path: Path) -> str | None:
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            texts = [t for p in pdf.pages if (t := p.extract_text())]
        return "\n\n".join(texts) if texts else None
    except Exception:
        return None


def _count_pages(path: Path) -> int:
    try:
        import pymupdf
        with pymupdf.open(path) as doc:
            return doc.page_count
    except Exception:
        return 0


# Trigrama letra-dígito-letra sin espacios: firma de fuente con ToUnicode roto
# (sustituye letras por dígitos, p.ej. "dicta11" en vez de "dictan").
_CORRUPT_TRIGRAM_RE = re.compile(r"[A-Za-zÁÉÍÓÚáéíóúÑñ][0-9]{1,2}[A-Za-zÁÉÍÓÚáéíóúÑñ]")


def looks_corrupt(text: str, threshold: int = 6) -> bool:
    """True si la capa de texto tiene corrupción sistemática de fuente.

    Solo cuenta trigramas repetidos ≥3 veces, para no confundir ruido disperso
    de tablas/códigos con una fuente realmente rota (los limpios dan 0).
    """
    counts = Counter(m.lower() for m in _CORRUPT_TRIGRAM_RE.findall(text))
    return sum(v for v in counts.values() if v >= 3) >= threshold


def extract(pdf_path: str | Path) -> ExtractedDocument:
    """Extrae un PDF digital a Markdown; marca ``corrupt`` si la fuente está rota."""
    path = Path(pdf_path)
    text = _try_pymupdf4llm(path)
    method = "pymupdf4llm"
    if not text:
        text, method = _try_pdfplumber(path) or "", "pdfplumber"
    return ExtractedDocument(
        filename=path.name, filepath=str(path.resolve()),
        text=text, pages=_count_pages(path), method=method,
        corrupt=looks_corrupt(text),
    )


def is_likely_digital(pdf_path: str | Path, sample_pages: int = 3) -> bool:
    """Heurística: ≥50 caracteres por página en promedio -> es digital."""
    try:
        import pdfplumber
        with pdfplumber.open(Path(pdf_path)) as pdf:
            pages = pdf.pages[:sample_pages]
            if not pages:
                return False
            return sum(len(p.extract_text() or "") for p in pages) / len(pages) >= 50
    except Exception:
        return False
