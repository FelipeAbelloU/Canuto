"""Genera pares pregunta-respuesta desde texto normativo para fine-tuning.

Formatos de salida:
  alpaca:   {"instruction": ..., "input": ..., "output": ...}
  sharegpt: {"conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class QAPair:
    question: str
    answer: str
    source: str = ""


def to_alpaca(pair: QAPair) -> dict:
    return {
        "instruction": pair.question,
        "input": "",
        "output": pair.answer,
    }


def to_sharegpt(pair: QAPair) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": pair.question},
            {"from": "gpt", "value": pair.answer},
        ]
    }


# ---------------------------------------------------------------------------
# Detección del tipo y referencia del documento
# ---------------------------------------------------------------------------

def _detect_doc_type(name: str) -> str:
    n = name.upper()
    if "R.C.F." in n or "RCF" in n:
        return "Resolución del Consejo de Facultad"
    if "RESOLUCIÓN RECTORAL" in n or "RESOLUCION RECTORAL" in n:
        return "Resolución Rectoral"
    if re.search(r"\bRR\b", n):
        return "Resolución Rectoral"
    if "RESOLUCIÓN ACADÉMICA" in n or "RESOLUCION ACADEMICA" in n:
        return "Resolución Académica"
    if "RESOLUCIÓN SUPERIOR" in n or "RESOLUCION SUPERIOR" in n:
        return "Resolución Superior"
    if "RESOLUCION" in n or "RESOLUCIÓN" in n:
        return "Resolución"
    if "ACUERDO ACADÉMICO" in n or "ACUERDO ACADEMICO" in n:
        return "Acuerdo Académico"
    if "ACUERDO SUPERIOR" in n:
        return "Acuerdo Superior"
    if "ACUERDO" in n:
        return "Acuerdo"
    if "CONVOCATORIA" in n:
        return "Convocatoria"
    return "documento normativo"


_DOC_ARTICLE = {
    "Resolución Rectoral": ("la", "La"),
    "Resolución Académica": ("la", "La"),
    "Resolución Superior": ("la", "La"),
    "Resolución": ("la", "La"),
    "Resolución del Consejo de Facultad": ("la", "La"),
    "Acuerdo Académico": ("el", "El"),
    "Acuerdo Superior": ("el", "El"),
    "Acuerdo": ("el", "El"),
    "Convocatoria": ("la", "La"),
    "documento normativo": ("el", "El"),
}


def _extract_ref_from_filename(name: str) -> str:
    """Extrae número y año del nombre del archivo (sin repetir el tipo de doc)."""
    # Buscar patrón "NÚMERO de AÑO" o "NÚMERO DE AÑO"
    match = re.search(r"N[°º]?\s*(\d{2,4})[^0-9]+?(20\d{2})", name, re.IGNORECASE)
    if match:
        return f"N° {match.group(1)} de {match.group(2)}"
    # Buscar solo número y año separados
    match = re.search(r"(\d{3,4})\s*[_\-\s]+\s*(20\d{2})", name)
    if match:
        return f"N° {match.group(1)} de {match.group(2)}"
    # Buscar RR_NNNN_de_AAAA o similar
    match = re.search(r"(\d{4})[_\s]+(20\d{2})", name)
    if match:
        return f"N° {match.group(1)} de {match.group(2)}"
    # Fallback: limpiar prefijos conocidos del nombre
    clean = re.sub(
        r"^(RESOLUCION|RESOLUCIÓN|ACUERDO|CONVOCATORIA)\s+"
        r"(ACADEMICO|ACADÉMICO|ACADEMICA|ACADÉMICA|RECTORAL|SUPERIOR)?\s*",
        "", name, flags=re.IGNORECASE
    ).replace("_", " ").replace("-", " ").strip()
    return clean[:60]


def _extract_title_from_text(text: str) -> str:
    """Extrae el título descriptivo del documento desde el texto.

    Los documentos de Unillanos incluyen el título entre comillas:
    '"Por la cual se establece..."'
    """
    match = re.search(r'[""«]([^""»\n]{30,300})[""»]', text[:2000])
    if match:
        return match.group(1).strip()
    # Fallback: buscar después de la fecha
    match = re.search(
        r'\(\w+\s+\d{1,2}\)\s*[""«]?(.{30,250}?)[""»\n]', text[:2000]
    )
    if match:
        return match.group(1).strip()
    return ""


def _extract_articles(text: str) -> list[tuple[str, str, str]]:
    """Extrae artículos del cuerpo normativo (después de RESUELVE/ACUERDA).

    Retorna lista de (número, título_si_existe, contenido).
    Maneja variantes OCR: ARTÍCULO, ARTICULO, ARTÉCULO.
    """
    # Trabajar solo desde la sección RESUELVE/ACUERDA para evitar capturar
    # artículos citados en el CONSIDERANDO (que son de otros documentos)
    for marker in ["RESUELVE", "ACUERDA", "ESTABLECE"]:
        idx = text.upper().find(marker)
        if idx != -1:
            text = text[idx:]
            break

    # Patrón flexible para artículos con posible título antes del punto
    pattern = re.compile(
        r"ART[ÍIÉ]CULO\s+(\w+)[°\.]?\s*[-–.]?\s*(.*?)(?=ART[ÍIÉ]CULO|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    results = []
    for m in pattern.finditer(text):
        num = m.group(1).strip()
        content = m.group(2).strip()
        if len(content) < 30:
            continue
        # Separar título del contenido si existe (primer punto o dos puntos)
        title_match = re.match(r"^([^.:\n]{5,80})[.:](.+)", content, re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
            body = title_match.group(2).strip()
        else:
            title = ""
            body = content
        # Limpiar contenido: máximo 600 chars para respuesta
        body = re.sub(r"\s+", " ", body)[:600].strip()
        results.append((num, title, body))
    return results


def _extract_paragrafos(text: str) -> list[tuple[str, str]]:
    """Extrae parágrafos (cláusulas adicionales de los artículos)."""
    pattern = re.compile(
        r"PARÁGRAFO\s*(\w*)[°\.]?\s*[-–.]?\s*(.*?)(?=PARÁGRAFO|ART[ÍIÉ]CULO|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    results = []
    for m in pattern.finditer(text):
        num = m.group(1).strip() or "único"
        content = re.sub(r"\s+", " ", m.group(2).strip())[:500]
        if len(content) > 40:
            results.append((num, content))
    return results


def _extract_resuelve_block(text: str) -> str:
    """Extrae el bloque RESUELVE/ACUERDA completo."""
    for keyword in ["RESUELVE", "ACUERDA", "ESTABLECE"]:
        pattern = rf"{keyword}[:\s]+([\s\S]{{100,2000}}?)(?:COMUNÍQUESE|CÚMPLASE|DADO|NOTIFÍQUESE|$)"
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return keyword, re.sub(r"\s+", " ", m.group(1).strip())[:1500]
    return "", ""


# ---------------------------------------------------------------------------
# Generador heurístico principal
# ---------------------------------------------------------------------------

def generate_heuristic(text: str, doc_name: str) -> list[QAPair]:
    """Genera pares QA automáticamente desde texto normativo.

    Produce preguntas variadas: resumen, artículos, parágrafos, vigencia.
    """
    pairs: list[QAPair] = []
    doc_type = _detect_doc_type(doc_name)
    ref = _extract_ref_from_filename(doc_name)
    title = _extract_title_from_text(text)

    # Descripción legible del documento
    doc_label = f"{doc_type} {ref}"
    if title:
        doc_label_full = f"{doc_type} {ref} — \"{title}\""
    else:
        doc_label_full = doc_label

    art_lower, art_upper = _DOC_ARTICLE.get(doc_type, ("el", "El"))
    intro = f"Según {art_lower} {doc_label} de la Universidad de los Llanos (Unillanos)"

    # 1. Pregunta general de qué trata el documento
    resumen_text = text[:1200].strip()
    pairs.append(QAPair(
        question=f"¿De qué trata {art_lower} {doc_label} de Unillanos?",
        answer=(
            f"{art_upper} {doc_label_full} de la Universidad de los Llanos trata sobre:\n\n"
            f"{resumen_text}"
        ),
        source=doc_name,
    ))

    # 2. Pregunta usando el título descriptivo del documento
    if title and len(title) > 20:
        pairs.append(QAPair(
            question=f"¿Qué normativa regula \"{title.lower()}\" en Unillanos?",
            answer=(
                f"Este tema está regulado por {art_lower} {doc_label_full} de la "
                f"Universidad de los Llanos.\n\n{resumen_text}"
            ),
            source=doc_name,
        ))

    # 3. Bloque RESUELVE / ACUERDA
    keyword, resuelve_body = _extract_resuelve_block(text)
    if resuelve_body:
        verb_map = {
            "RESUELVE": "resuelve",
            "ACUERDA": "establece mediante acuerdo",
            "ESTABLECE": "establece",
        }
        verb = verb_map.get(keyword, "dispone")
        pairs.append(QAPair(
            question=f"¿Qué {verb} {art_lower} {doc_label}?",
            answer=f"{intro}, la norma {verb}:\n\n{resuelve_body}",
            source=doc_name,
        ))

    # 4. Artículos individuales (máximo 8)
    articles = _extract_articles(text)
    for num, art_title, content in articles[:8]:
        if art_title:
            q = f"¿Qué establece el artículo {num} ({art_title}) de la {doc_label}?"
            a = (
                f"El artículo {num} de la {doc_label}, referente a {art_title}, "
                f"establece:\n\n{content}"
            )
        else:
            q = f"¿Qué dice el artículo {num} de la {doc_label}?"
            a = f"El artículo {num} de la {doc_label} establece:\n\n{content}"
        pairs.append(QAPair(question=q, answer=a, source=doc_name))

    # 5. Parágrafos importantes (máximo 3)
    paragrafos = _extract_paragrafos(text)
    for num, content in paragrafos[:3]:
        pairs.append(QAPair(
            question=f"¿Qué indica el parágrafo {num} de {art_lower} {doc_label}?",
            answer=(
                f"El parágrafo {num} de {art_lower} {doc_label} establece:\n\n{content}"
            ),
            source=doc_name,
        ))

    # 6. Vigencia del documento
    vigencia_match = re.search(
        r"(rige\s+a\s+partir[^.]{10,200}|vigencia[^.]{10,200})",
        text,
        re.IGNORECASE,
    )
    if vigencia_match:
        pairs.append(QAPair(
            question=f"¿Desde cuándo rige {art_lower} {doc_label}?",
            answer=(
                f"{intro}, la vigencia del documento indica que: "
                f"{vigencia_match.group(0).strip()}"
            ),
            source=doc_name,
        ))

    return pairs


def generate_template(doc_name: str, n: int = 5) -> list[QAPair]:
    """Genera plantillas vacías para completar manualmente."""
    doc_type = _detect_doc_type(doc_name)
    ref = _extract_ref_from_filename(doc_name)
    return [
        QAPair(
            question=f"[PREGUNTA {i+1} sobre {doc_type} {ref}]",
            answer=f"[RESPUESTA {i+1} — completar manualmente con información de {doc_name}]",
            source=doc_name,
        )
        for i in range(n)
    ]


def save_dataset(
    pairs: list[QAPair],
    output_path: str | Path,
    fmt: str = "alpaca",
) -> None:
    """Guarda el dataset en formato JSON alpaca o sharegpt."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converter = to_alpaca if fmt == "alpaca" else to_sharegpt
    entries = [converter(p) for p in pairs]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"  Dataset guardado: {output_path} ({len(entries)} entradas)")
