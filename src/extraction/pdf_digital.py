"""Extrae texto de PDFs con texto seleccionable (no escaneados).
Usa pdfplumber como extractor principal y pypdf como fallback.
"""
from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ExtractedDocument:
    filename: str
    filepath: str
    text: str
    pages: int
    method: str
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)

    def is_empty(self) -> bool:
        return self.char_count < 100


def _try_pdfplumber(path: Path) -> str | None:
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
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return len(pdf.pages)
    except Exception:
        try:
            from pypdf import PdfReader
            return len(PdfReader(str(path)).pages)
        except Exception:
            return 0


def extract(pdf_path: str | Path) -> ExtractedDocument:
    """Extrae texto de un PDF digital. Retorna ExtractedDocument (texto vacío si falla)."""
    path = Path(pdf_path)
    pages = _count_pages(path)

    text = _try_pdfplumber(path)
    method = "pdfplumber"

    if not text:
        text = _try_pypdf(path)
        method = "pypdf"

    return ExtractedDocument(
        filename=path.name,
        filepath=str(path.resolve()),
        text=text or "",
        pages=pages,
        method=method,
    )


def is_likely_digital(pdf_path: str | Path, sample_pages: int = 3) -> bool:
    """Heurística: si se extraen ≥50 caracteres por página en promedio, el PDF es digital."""
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
