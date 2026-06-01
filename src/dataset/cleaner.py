"""Limpia y normaliza texto extraído de PDFs normativos de Unillanos."""
from __future__ import annotations

import re
from collections import Counter


def _remove_repeated_headers(text: str) -> str:
    """Elimina encabezados de página que el OCR repite en cada hoja.

    En documentos escaneados de Unillanos, cada página comienza con el mismo
    bloque institucional. Se detecta por frecuencia de aparición.
    """
    lines = text.split("\n")
    # Contar cuántas veces aparece cada línea no vacía
    counts = Counter(ln.strip() for ln in lines if len(ln.strip()) > 20)
    # Líneas que aparecen 3+ veces son probablemente encabezados repetidos
    repeated = {ln for ln, cnt in counts.items() if cnt >= 3}
    if not repeated:
        return text
    cleaned = [ln for ln in lines if ln.strip() not in repeated]
    return "\n".join(cleaned)


def _fix_ocr_substitutions(text: str) -> str:
    """Corrige sustituciones comunes del OCR en documentos institucionales."""
    # N' → N° (apóstrofe mal reconocido como símbolo de grado)
    text = re.sub(r"\bN['`']\s*(\d)", r"N° \1", text)
    # ARTÉCULO → ARTÍCULO (É mal reconocida)
    text = re.sub(r"\bART[ÉE]CULO\b", "ARTÍCULO", text, flags=re.IGNORECASE)
    # Nº → N° (normalización)
    text = re.sub(r"\bN[oº°]\s*\.?\s*(\d)", r"N° \1", text)
    # Separar palabras pegadas antes de mayúsculas en contextos normativos
    # e.g. "ArtículoCuarto" → no se toca, pero "delosProgramas" → no aplicar
    return text


def clean(text: str) -> str:
    """Limpieza completa para texto extraído de documentos normativos."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Correcciones OCR
    text = _fix_ocr_substitutions(text)

    # Remover guiones de silabeo al final de línea
    text = re.sub(r"-\n([a-záéíóúüñ])", r"\1", text, flags=re.IGNORECASE)

    # Unir líneas que no terminan párrafo
    text = re.sub(r"(?<![.;:?!\n])\n(?![A-ZÁÉÍÓÚÜÑ\n\d•\-])", " ", text)

    # Eliminar espacios múltiples
    text = re.sub(r" {2,}", " ", text)

    # Eliminar líneas muy cortas (artefactos OCR, números de página sueltos)
    lines = [ln for ln in text.split("\n") if len(ln.strip()) > 3 or ln.strip() == ""]
    text = "\n".join(lines)

    # Eliminar encabezados repetidos de página
    text = _remove_repeated_headers(text)

    # Colapsar líneas en blanco repetidas
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
