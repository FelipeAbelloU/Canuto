"""Genera pares pregunta-respuesta desde documentos normativos en Markdown.

Lee el .md estructurado que produce extract_text.py y aprovecha:
  - el frontmatter (tipo, número, año, título) → citas exactas sin adivinar,
  - los encabezados (## CONSIDERANDO/RESUELVE, ### ARTÍCULO N, #### PARÁGRAFO)
    → contenido real de cada artículo, sin regex frágiles sobre texto crudo.

Las respuestas son el texto real del documento con su cita, para que el modelo
aprenda hechos fundamentados y no invente.

Formatos de salida:
  alpaca:   {"instruction": ..., "input": ..., "output": ...}
  sharegpt: {"conversations": [{"from": "human", ...}, {"from": "gpt", ...}]}
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
    return {"instruction": pair.question, "input": "", "output": pair.answer}


def to_sharegpt(pair: QAPair) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": pair.question},
            {"from": "gpt", "value": pair.answer},
        ]
    }


# ---------------------------------------------------------------------------
# Tipos de documento que NO generan buenos QA (tablas, mallas, planes de estudio)
# ---------------------------------------------------------------------------
_SKIP_TYPES = {
    "MALLA_CURRICULAR", "Malla_curricular",
    "PLAN_DE_ESTUDIOS_DEL_PROGRAMA", "Plan_de_estudios_del_programa",
    "Contenido_Programatico", "Plan",
}


def _skip_type(tipo: str) -> bool:
    # Estos tipos son tablas: el texto plano no produce preguntas útiles.
    return tipo in _SKIP_TYPES


# ---------------------------------------------------------------------------
# Concordancia gramatical según el tipo (la Resolución / el Acuerdo)
# ---------------------------------------------------------------------------
def _gender(tipo_legible: str) -> tuple[str, str, str]:
    """Devuelve (artículo, artículo_mayúscula, contracción) para el tipo.

    'la Resolución / La / de la'  ·  'el Acuerdo / El / del'
    """
    t = tipo_legible.lower()
    # Acta es femenino pero lleva artículo masculino ("el acta", "del acta").
    if t.startswith(("acuerdo", "decreto", "acta", "documento", "reglamento", "plan")):
        return ("el", "El", "del")
    return ("la", "La", "de la")


# Tipos que son legislación NACIONAL, no normativa propia de Unillanos:
# se citan sin "de Unillanos" para no atribuirle autoría equivocada.
_NATIONAL_HINTS = ("NACIONAL", "MINISTERIO", "_MEN", "CONPES", "LEY", "ORDENANZA", "CESU")


def _is_national(tipo: str) -> bool:
    u = tipo.upper()
    return u == "LEY" or any(h in u for h in _NATIONAL_HINTS)


# Un número de artículo válido: arábigo, romano u ordinal escrito.
_ROMAN_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
_ORDINAL_WORDS = {
    "primero", "segundo", "tercero", "cuarto", "quinto", "sexto", "séptimo",
    "septimo", "octavo", "noveno", "décimo", "decimo", "único", "unico",
}


def _valid_num(n: str) -> str:
    """Devuelve el número si es válido (arábigo/romano/ordinal), o '' si no."""
    n = n.strip()
    if re.fullmatch(r"\d{1,3}", n):
        return n
    if _ROMAN_RE.match(n) and len(n) <= 5:
        return n.upper()
    if n.lower() in _ORDINAL_WORDS:
        return n.lower()
    return ""


# ---------------------------------------------------------------------------
# Parser de la estructura Markdown del documento
# ---------------------------------------------------------------------------
_H2_RE = re.compile(r"^##\s+(.+)$")
_ART_RE = re.compile(r"^###\s+ART[IÍ]CULO\s*(.*)$", re.IGNORECASE)   # tolera "ARTICULO" del OCR
_PAR_RE = re.compile(r"^####\s+PAR[AÁ]GRAFO\s*(.*)$", re.IGNORECASE)


@dataclass
class DocStructure:
    descripcion: str = ""                       # el "Por la cual…" del encabezado
    considerando: str = ""                      # texto bajo ## CONSIDERANDO
    resuelve: str = ""                           # texto bajo ## RESUELVE antes del 1er artículo
    articles: list[tuple[str, str]] = field(default_factory=list)      # (número, contenido)


def _join(lines: list[str]) -> str:
    # Une las líneas de un bloque en un párrafo limpio.
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _clean_art_num(raw: str) -> str:
    # Limpia el número del encabezado "### ARTÍCULO 5°." -> "5".
    return raw.strip().strip(".:°º-–) ").strip()


def _parse_structure(body_md: str) -> DocStructure:
    """Recorre el Markdown por encabezados y separa encabezado, secciones y artículos."""
    lines = body_md.splitlines()

    header: list[str] = []          # texto antes del primer ##
    sections: dict[str, list[str]] = {}
    articles: list[list] = []       # [num, [líneas]]
    target = header                 # dónde se acumulan las líneas de contenido actuales

    for raw in lines:
        s = raw.rstrip()

        # Título del documento (# ...): se ignora, ya está en el frontmatter.
        if s.startswith("# ") and not s.startswith("## "):
            continue

        # Artículo (### ARTÍCULO N) -> nuevo bloque de artículo.
        m_art = _ART_RE.match(s)
        if m_art:
            articles.append([_clean_art_num(m_art.group(1)), []])
            target = articles[-1][1]
            continue

        # Parágrafo (#### PARÁGRAFO): pertenece al artículo actual, así que su
        # contenido se pliega dentro del bloque vigente con una etiqueta.
        m_par = _PAR_RE.match(s)
        if m_par:
            num_par = m_par.group(1).strip()
            target.append(f"Parágrafo{(' ' + num_par) if num_par else ''}:")
            continue

        # Sección (## CONSIDERANDO / RESUELVE / CAPÍTULO ...).
        m_h2 = _H2_RE.match(s)
        if m_h2 and not s.startswith("### "):
            name = m_h2.group(1).strip().upper()
            sections.setdefault(name, [])
            target = sections[name]
            continue

        # Línea de contenido normal.
        target.append(s)

    st = DocStructure()
    st.descripcion = _extract_descripcion(header)
    st.considerando = _join(sections.get("CONSIDERANDO", []))
    st.resuelve = _join(sections.get("RESUELVE", []) or sections.get("ACUERDA", []))
    st.articles = [(num, _join(ls)) for num, ls in articles if _join(ls)]
    return st


def _extract_descripcion(header_lines: list[str]) -> str:
    """Saca el 'Por la cual…' entre comillas del encabezado."""
    header = " ".join(header_lines)
    m = re.search(r'[“"«]\s*(Por\s[^”"»]{15,400})[”"»]', header, re.IGNORECASE)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


# ---------------------------------------------------------------------------
# Generador principal
# ---------------------------------------------------------------------------
def _doc_ref(meta: dict) -> tuple[str, str, str]:
    """Devuelve (tipo_legible, cita_corta, frase_completa) desde el frontmatter."""
    tipo = meta.get("tipo_legible") or meta.get("tipo", "") or "documento normativo"
    numero = meta.get("numero", "")
    anio = meta.get("anio", "")
    if numero and anio:
        cita = f"{tipo} No. {numero} de {anio}"
    elif anio:
        cita = f"{tipo} ({anio})"
    else:
        cita = tipo
    return tipo, cita, f"{cita} de la Universidad de los Llanos (Unillanos)"


def generate_heuristic(body_md: str, meta: dict, max_pairs: int = 10) -> list[QAPair]:
    """Genera ~max_pairs pares QA fundamentados desde un documento en Markdown.

    Estrategia:
      1. Pregunta(s) general(es) sobre de qué trata el documento.
      2. Una pregunta por artículo, con el texto real del artículo como respuesta.
      3. Fallback (documentos sin artículos): usar CONSIDERANDO/RESUELVE.
    Todas las respuestas citan el documento por su número y año exactos.
    """
    if _skip_type(meta.get("tipo", "")):
        return []

    tipo, cita, _ = _doc_ref(meta)
    source = meta.get("fuente") or cita
    la, La, de_la = _gender(tipo)

    # Legislación nacional: se cita sin atribuirla a Unillanos.
    nacional = _is_national(meta.get("tipo", ""))
    suf_q = "" if nacional else " de Unillanos"                       # sufijo de la pregunta
    entidad = "" if nacional else " de la Universidad de los Llanos (Unillanos)"  # en la respuesta

    st = _parse_structure(body_md)
    pairs: list[QAPair] = []

    # ── 1. Preguntas generales ────────────────────────────────────────────
    if st.descripcion:
        intro_ans = f"{La} {cita}{entidad} es la norma «{st.descripcion}»."
        # Añadir el primer artículo como contexto de lo que dispone.
        if st.articles:
            intro_ans += f" En su artículo {st.articles[0][0] or '1'} establece: {st.articles[0][1]}"
        pairs.append(QAPair(f"¿De qué trata {la} {cita}{suf_q}?", intro_ans, source))
        pairs.append(QAPair(f"¿Qué regula {la} {cita}{suf_q}?", intro_ans, source))

    # ── 2. Un par por artículo (el núcleo del dataset) ────────────────────
    # Se garantiza un número ÚNICO por artículo para no crear preguntas
    # idénticas con respuestas distintas (datos contradictorios).
    seen: set[str] = set()
    for idx, (raw_num, content) in enumerate(st.articles, start=1):
        if len(pairs) >= max_pairs - 1:   # dejar sitio para la vigencia
            break
        num = _valid_num(raw_num)
        if not num or num in seen:
            # Si el número no es válido o se repite: usar el número que
            # encabeza el contenido ("1. OTORGAR...") o la posición.
            lead = re.match(r"^\s*(\d{1,3})[.\)]", content)
            num = lead.group(1) if (lead and lead.group(1) not in seen) else str(idx)
        if num in seen:
            continue
        seen.add(num)

        q = f"¿Qué establece el artículo {num} {de_la} {cita}{suf_q}?"
        a = f"Según el artículo {num} {de_la} {cita}{entidad}, se establece: {content}"
        pairs.append(QAPair(q, a, source))

    # ── 3. Fallback: documentos sin artículos (actas, comunicados) ────────
    if not st.articles:
        cuerpo = st.resuelve or st.considerando or st.descripcion
        if cuerpo:
            if not st.descripcion:
                pairs.append(QAPair(
                    f"¿Qué dispone {la} {cita}{suf_q}?",
                    f"{La} {cita}{entidad} dispone: {cuerpo[:1200]}",
                    source,
                ))
            if st.considerando:
                pairs.append(QAPair(
                    f"¿Cuál es el fundamento {de_la} {cita}{suf_q}?",
                    f"{La} {cita}{entidad} considera: {st.considerando[:1200]}",
                    source,
                ))

    # ── 4. Vigencia (si aparece explícita) ────────────────────────────────
    vig = re.search(r"(rige\s+a\s+partir[^.\n]{5,160}|entrar[áa]?\s+en\s+vigen[^.\n]{5,160})",
                    body_md, re.IGNORECASE)
    if vig and len(pairs) < max_pairs:
        pairs.append(QAPair(
            f"¿Desde cuándo rige {la} {cita}{suf_q}?",
            f"{La} {cita}{entidad} {vig.group(1).strip()}.",
            source,
        ))

    return pairs[:max_pairs]


def generate_template(meta: dict, n: int = 5) -> list[QAPair]:
    """Genera plantillas vacías para completar manualmente."""
    if _skip_type(meta.get("tipo", "")):
        return []
    _, cita, _ = _doc_ref(meta)
    source = meta.get("fuente") or cita
    return [
        QAPair(
            question=f"[PREGUNTA {i + 1} sobre {cita}]",
            answer=f"[RESPUESTA {i + 1} — completar con información de {cita}]",
            source=source,
        )
        for i in range(n)
    ]


def save_dataset(pairs: list[QAPair], output_path: str | Path, fmt: str = "alpaca") -> None:
    """Guarda el dataset en formato JSON alpaca o sharegpt."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter = to_alpaca if fmt == "alpaca" else to_sharegpt
    entries = [converter(p) for p in pairs]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"  Dataset guardado: {output_path} ({len(entries)} entradas)")
