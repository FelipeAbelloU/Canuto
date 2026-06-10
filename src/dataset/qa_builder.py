"""Genera pares pregunta-respuesta desde texto normativo para fine-tuning.

Formatos de salida:
  alpaca:   {"instruction": ..., "input": ..., "output": ...}
  sharegpt: {"conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from dataclasses import dataclass


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
# Archivos que NO deben generar pares QA (no son normativa académica)
# ---------------------------------------------------------------------------

_SKIP_STEMS = {
    # Correspondencia administrativa / radicados
    "2026-EE-01207",
    # Comunicados públicos (no normativa)
    "Comunicado a la opinión pública",
    "Comunicado a la opinón pública",
    # Conceptos jurídicos (no vinculantes)
    "Concepto sobre regulación protesta-Unillanos",
    # Respuestas de dependencias (no normativa)
    "DIV. FIN 042, 07-04-26 RESPUESTA AL CONS. ACADEMICO",
    # Informes de gestión (reportes, no normativa)
    "INFORME DE GESTION FCE",
    "Informe de Gestión No II P.A FCS 2019",
    # Manuales de instalaciones físicas
    "MANUAL DE INSTRUCCIONES DE PISCINA",
    # Exposición de motivos (documento interno previo a un acuerdo, no normativa)
    "Motivos",
    # Resoluciones sobre contratación institucional durante elecciones
    # (no relevante para consultas estudiantiles)
    "RESOLUCIÓN No. 0127-26 - Rectoria Universidad de los Llanos",
    "Resolución Rectoral No. 0158 de 2026 ",   # modifica la 0127
    "Resolución Rectoral No. 0158 de 2026",    # sin espacio final (por si acaso)
    # Convocatoria de representación estudiantil (proceso administrativo)
    "RESOLUCION RECTORAL 0809 DE 2026 ",
    "RESOLUCION RECTORAL 0809 DE 2026",
    # Borrador sin número de resolución real (tiene "XX" como placeholder)
    "Cooperativas",
    # Calendario académico 2020 — desactualizado
    "RESOLUCION ACADEMICA N° 041 DE 2020. CALENDARIO ACADEMICO2020.pago 24 de junio - DEFINITIVA V°B° (1)",
    # Duplicado — se usa la versión (2) con nombre más completo
    "Resolución Rectoral 068 de 2025 - Incapacidades",
}


def should_skip(stem: str) -> bool:
    """Retorna True si el archivo no debe generar pares QA."""
    return stem.strip() in _SKIP_STEMS


# ---------------------------------------------------------------------------
# Identificación y referencia del documento
# ---------------------------------------------------------------------------

def _detect_doc_type(name: str) -> str:
    n = name.upper()
    if "R.C.F." in n or "RCF" in n or "CF." in n:
        return "Resolución del Consejo de Facultad"
    if "RESOLUCIÓN RECTORAL" in n or "RESOLUCION RECTORAL" in n:
        return "Resolución Rectoral"
    if re.search(r"\bRR[\s_]", n):
        return "Resolución Rectoral"
    if "RESOLUCIÓN ACADÉMICA" in n or "RESOLUCION ACADEMICA" in n:
        return "Resolución Académica"
    if "RESOLUCIÓN SUPERIOR" in n or "RESOLUCION SUPERIOR" in n:
        return "Resolución Superior"
    if re.search(r"RESOLUCIÓN\s+SUPERIOR", n):
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
    if "RÉGIMEN" in n or "REGIMEN" in n:
        return "Reglamento"
    return "documento normativo"


# Género gramatical para concordancia (articulo, preposicion)
_DOC_GENDER = {
    "Resolución Rectoral": ("la", "La", "de la", "a la"),
    "Resolución Académica": ("la", "La", "de la", "a la"),
    "Resolución Superior": ("la", "La", "de la", "a la"),
    "Resolución": ("la", "La", "de la", "a la"),
    "Resolución del Consejo de Facultad": ("la", "La", "de la", "a la"),
    "Acuerdo Académico": ("el", "El", "del", "al"),
    "Acuerdo Superior": ("el", "El", "del", "al"),
    "Acuerdo": ("el", "El", "del", "al"),
    "Convocatoria": ("la", "La", "de la", "a la"),
    "Reglamento": ("el", "El", "del", "al"),
    "documento normativo": ("el", "El", "del", "al"),
}


def _extract_ref(name: str) -> str:
    """Extrae número y año del nombre del archivo para construir la referencia.

    Maneja los patrones comunes en los nombres de documentos de Unillanos:
    - "N° 074 de 2026", "N.074 de 2026" (con prefijo N°)
    - "074 de 2026" (sin prefijo, pero con 'de' escrito)
    - "0074_2026", "074-2026" (separados por guión o guión bajo)
    """
    # Patrón 1: N° NNN de AAAA  (con prefijo explícito)
    m = re.search(r"N[°oº]?\s*\.?\s*0*(\d{2,4})\s+(?:de|DE)\s+(20\d{2})", name, re.IGNORECASE)
    if m:
        return f"N° {m.group(1)} de {m.group(2)}"
    # Patrón 2: NNN de AAAA  (sin prefijo, número corto)
    m = re.search(r"\b0*(\d{2,4})\s+(?:de|DE)\s+(20\d{2})\b", name)
    if m:
        return f"N° {m.group(1)} de {m.group(2)}"
    # Patrón 3: NNNN_AAAA o NNNN-AAAA
    m = re.search(r"\b0*(\d{3,4})[_\-](20\d{2})\b", name)
    if m:
        return f"N° {m.group(1)} de {m.group(2)}"
    # Fallback: extraer solo el número si hay año en el nombre
    m = re.search(r"\b0*(\d{4})\b.*(20\d{2})", name)
    if m:
        return f"N° {m.group(1)} de {m.group(2)}"
    # Último recurso: limpiar prefijo del tipo de documento
    clean = re.sub(
        r"^(RESOLUCION|RESOLUCIÓN|ACUERDO|CONVOCATORIA|REGLAMENTO|RR|R\.C\.F\.)\s*"
        r"(ACADEMICO|ACADÉMICO|ACADEMICA|ACADÉMICA|RECTORAL|SUPERIOR|CF|No\.?)?\s*",
        "", name, flags=re.IGNORECASE
    ).strip()
    return clean[:60]


def _extract_title(text: str) -> str:
    """Extrae el título descriptivo entre comillas del encabezado del documento.

    Busca solo en la sección del encabezado (antes del CONSIDERANDO) para
    evitar capturar citas de leyes externas que aparecen en el cuerpo.
    """
    # Limitar al encabezado: antes de CONSIDERANDO o primeros 2500 chars
    header_end = text.upper().find("CONSIDERANDO")
    header = text[:header_end] if header_end > 0 else text[:2500]

    # Título típico: entre comillas, empieza con "Por" o "Por medio"
    candidates = re.findall(r'[""«]([^""»\n]{25,400})[""»]', header)
    for c in candidates:
        c = c.strip()
        # El título real del documento suele empezar con "Por"
        if re.match(r'^Por\b', c, re.IGNORECASE):
            # Descartar títulos que solo citan leyes externas
            if not re.search(r'^Por (la cual|la que|el cual) se organiza el servicio', c, re.IGNORECASE):
                return c
    # Fallback: primer candidato entre comillas
    if candidates:
        return candidates[0].strip()
    return ""


# ---------------------------------------------------------------------------
# Extracción de artículos y parágrafos
# ---------------------------------------------------------------------------

# Palabras que NO son números de artículo — artefactos del OCR o texto normativo
_INVALID_ART_NUMS = {
    "anterior", "siguiente", "presente", "citado", "mencionado",
    "referido", "transcrito", "precitado", "dicho", "arriba", "atrás",
}


def _extract_articles(text: str) -> list[tuple[str, str, str]]:
    """Extrae artículos desde el bloque RESUELVE/ACUERDA.

    Retorna lista de (número_ordinal, título_si_existe, contenido).
    """
    # Buscar desde RESUELVE/ACUERDA para evitar artículos del CONSIDERANDO
    body = text
    for marker in ["RESUELVE", "ACUERDA", "ESTABLECE", "DISPONE"]:
        idx = text.upper().find(marker)
        if idx != -1:
            body = text[idx:]
            break

    pattern = re.compile(
        r"ART[ÍIÉiié]CULO\s+([\w°]+)[°\.]?\s*[-–.]?\s*(.*?)(?=ART[ÍIÉiié]CULO\s+[\w°]+|\Z)",
        re.IGNORECASE | re.DOTALL,
    )

    results = []
    for m in pattern.finditer(body):
        raw_num = m.group(1).strip()
        content = m.group(2).strip()

        # Descartar si el "número" es una palabra relativa (artefacto OCR)
        if raw_num.lower() in _INVALID_ART_NUMS:
            continue

        if len(content) < 25:
            continue

        # Convertir número ordinal textual a arábigo si aplica
        num = _ordinal_to_arabic(raw_num) or raw_num

        # Separar título del cuerpo (primer punto o dos puntos en la primera línea)
        title_m = re.match(r"^([^.:\n]{4,80})[.:](.+)", content, re.DOTALL)
        if title_m:
            title = title_m.group(1).strip()
            body_text = title_m.group(2).strip()
            # Rechazar títulos que parezcan texto continuo (más de 5 palabras con minúsculas)
            if sum(1 for w in title.split() if w[0].islower()) > 3:
                title = ""
                body_text = content
        else:
            title = ""
            body_text = content

        # Limpiar artefactos OCR comunes
        body_text = re.sub(r"\s+", " ", body_text)

        # Filtrar artefactos que vienen del CONSIDERANDO
        if re.search(r"artículo anterior|art[íi]culo que antecede|el anterior artículo", title, re.IGNORECASE):
            continue
        if re.search(r"artículo anterior|art[íi]culo que antecede", body_text[:80], re.IGNORECASE):
            continue

        # Limitar longitud del cuerpo a 700 chars
        body_text = body_text[:700].strip()

        results.append((num, title, body_text))

    return results


_ORDINALS_ES = {
    "primero": "1", "segundo": "2", "tercero": "3", "cuarto": "4",
    "quinto": "5", "sexto": "6", "séptimo": "7", "sétimo": "7",
    "octavo": "8", "noveno": "9", "décimo": "10",
    "primero.": "1", "único": "único",
}


def _ordinal_to_arabic(word: str) -> str:
    return _ORDINALS_ES.get(word.lower().rstrip("."), "")


def _extract_paragrafos(text: str) -> list[tuple[str, str]]:
    """Extrae parágrafos del documento."""
    pattern = re.compile(
        r"PAR[ÁA]GRAFO\s*(\w*)[°\.]?\s*[-–.]?\s*(.*?)(?=PAR[ÁA]GRAFO|ART[ÍIÉiié]CULO|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    results = []
    for m in pattern.finditer(text):
        num = m.group(1).strip().capitalize() or "único"
        content = re.sub(r"\s+", " ", m.group(2).strip())[:500]
        if len(content) > 40:
            results.append((num, content))
    return results


def _extract_resuelve(text: str) -> tuple[str, str]:
    """Extrae el bloque completo RESUELVE/ACUERDA."""
    for kw in ["RESUELVE", "ACUERDA", "ESTABLECE"]:
        m = re.search(
            rf"{kw}[:\s]+([\s\S]{{80,2000}}?)(?:COMUNÍQUESE|CÚMPLASE|DADO|NOTIFÍQUESE|PUBLÍQUESE|$)",
            text, re.IGNORECASE,
        )
        if m:
            return kw, re.sub(r"\s+", " ", m.group(1).strip())[:1500]
    return "", ""


# ---------------------------------------------------------------------------
# Generación de variantes de pregunta según tipo de artículo
# ---------------------------------------------------------------------------

_KEYWORD_QUESTIONS = [
    # (patrón_en_título, [(pregunta_template, ), ...])
    # Los templates usan {topic} = título del artículo en minúsculas
    (r"requisito|exig|condicion|condición",
     ["¿Cuáles son los requisitos para {topic} según Unillanos?",
      "¿Qué condiciones debo cumplir para {topic} en Unillanos?",
      "¿Qué se necesita para {topic} en la Universidad de los Llanos?"]),

    (r"procedimiento|proceso|pasos|trámite|trámite",
     ["¿Cómo se realiza {topic} en Unillanos?",
      "¿Cuáles son los pasos para {topic} en Unillanos?",
      "¿Qué debo hacer para {topic} según el reglamento de Unillanos?"]),

    (r"plazo|término|días|fecha|tiempo",
     ["¿Cuánto tiempo tengo para {topic} en Unillanos?",
      "¿Cuál es el plazo para {topic} según la normativa de Unillanos?"]),

    (r"apoyo|beneficio|subsidio|estímulo|descuento|beca",
     ["¿A qué apoyo económico puedo acceder para {topic} en Unillanos?",
      "¿Qué beneficios ofrece Unillanos relacionados con {topic}?",
      "¿Cuáles son los estímulos disponibles para {topic} en Unillanos?"]),

    (r"sanción|falta|infracción|disciplin",
     ["¿Cuáles son las sanciones por {topic} en Unillanos?",
      "¿Qué consecuencias tiene {topic} según el reglamento de Unillanos?"]),

    (r"exenc|excepción|exonera|exempt",
     ["¿Quiénes están exentos de {topic} en Unillanos?",
      "¿A quiénes no aplica {topic} según la normativa de Unillanos?"]),

    (r"definici|concepto|significad|entend",
     ["¿Qué significa {topic} en el contexto normativo de Unillanos?",
      "¿Cómo define Unillanos {topic}?"]),

    (r"objeto|objetivo|propósito|finalidad|alcance",
     ["¿Cuál es el objeto de {topic} en Unillanos?",
      "¿Cuál es el alcance de {topic} en la normativa de Unillanos?"]),

    (r"vigencia|rige|inicio|fecha",
     ["¿Desde cuándo rige {topic} en Unillanos?",
      "¿Cuándo entró en vigor {topic}?"]),
]


def _question_variants_for_article(
    art_num: str,
    art_title: str,
    doc_label: str,
) -> list[str]:
    """Genera 2-3 preguntas variadas para un artículo según su título."""
    if not art_title:
        return [f"¿Qué establece el artículo {art_num} {doc_label}?"]

    topic = art_title.lower().strip().rstrip(".")

    # Títulos genéricos de una sola palabra no producen buenas preguntas con templates
    # (ej: "alcance", "definiciones", "objeto" → preguntas redundantes)
    is_generic_single = (
        len(topic.split()) <= 2
        and topic in {
            "alcance", "objeto", "definiciones", "definición", "vigencia",
            "ámbito", "ambito", "aplicación", "aplicacion", "disposiciones",
            "generales", "publicidad",
        }
    )

    if not is_generic_single:
        for pattern, templates in _KEYWORD_QUESTIONS:
            if re.search(pattern, topic, re.IGNORECASE):
                questions = []
                for tmpl in templates[:2]:
                    questions.append(tmpl.format(topic=topic))
                # Siempre incluir una variante técnica explícita
                questions.append(
                    f"¿Qué establece el artículo {art_num} ({art_title}) "
                    f"{doc_label}?"
                )
                return questions

    # Fallback: variantes genéricas pero claras
    return [
        f"¿Qué establece el artículo {art_num} ({art_title}) {doc_label}?",
        f"¿Qué indica el artículo {art_num} sobre {topic} según Unillanos?",
    ]


# ---------------------------------------------------------------------------
# Generador heurístico principal
# ---------------------------------------------------------------------------

def generate_heuristic(text: str, doc_name: str) -> list[QAPair]:
    """Genera pares QA automáticamente desde texto normativo.

    Mejoras respecto a la versión anterior:
    - Excluye documentos no normativos via should_skip()
    - Múltiples variantes de pregunta por artículo (orientadas al estudiante)
    - Citas exactas con número de artículo en cada respuesta
    - Concordancia gramatical correcta (del/de la según el género del tipo doc)
    - No genera "artículo anterior" ni preguntas con artefactos de OCR
    """
    if should_skip(doc_name):
        return []

    pairs: list[QAPair] = []
    doc_type = _detect_doc_type(doc_name)
    ref = _extract_ref(doc_name)
    title = _extract_title(text)

    art_lo, art_hi, de_art, a_art = _DOC_GENDER.get(doc_type, ("el", "El", "del", "al"))

    doc_label_short = f"{doc_type} {ref}"
    doc_label_full = f"{doc_type} {ref} — \"{title}\"" if title else doc_label_short

    intro = f"Según {art_lo} {doc_label_short} de la Universidad de los Llanos (Unillanos)"
    resumen = text[:1400].strip()

    # ── 1. Pregunta general (3 variantes) ────────────────────────────────────
    general_qs = [
        f"¿De qué trata {art_lo} {doc_label_short} de Unillanos?",
        f"¿Qué regula {art_lo} {doc_label_short} en Unillanos?",
    ]
    if title and len(title) > 20:
        general_qs.append(f"¿Qué normativa establece \"{title.lower()}\" en Unillanos?")

    for q in general_qs:
        pairs.append(QAPair(
            question=q,
            answer=(
                f"{art_hi} {doc_label_full} de la Universidad de los Llanos "
                f"trata sobre:\n\n{resumen}"
            ),
            source=doc_name,
        ))

    # ── 2. Bloque RESUELVE / ACUERDA (2 variantes) ────────────────────────────
    kw, resuelve_body = _extract_resuelve(text)
    if resuelve_body:
        verb_map = {
            "RESUELVE": ("resuelve", "dispone"),
            "ACUERDA": ("acuerda", "establece"),
            "ESTABLECE": ("establece", "determina"),
        }
        v1, v2 = verb_map.get(kw, ("dispone", "establece"))
        pairs.append(QAPair(
            question=f"¿Qué {v1} {art_lo} {doc_label_short}?",
            answer=f"{intro}, la norma {v1}:\n\n{resuelve_body}",
            source=doc_name,
        ))
        pairs.append(QAPair(
            question=f"¿Qué {v2} {art_lo} {doc_label_short} de Unillanos?",
            answer=f"{intro}, la norma {v2}:\n\n{resuelve_body}",
            source=doc_name,
        ))

    # ── 3. Artículos individuales (máx. 10 artículos, 2-3 preguntas c/u) ─────
    articles = _extract_articles(text)
    for num, art_title, content in articles[:10]:
        cite = (
            f"Según el Artículo {num} {de_art} {doc_label_short} de Unillanos"
        )
        if art_title:
            answer = (
                f"{cite} (referente a {art_title}), se establece:\n\n{content}"
            )
        else:
            answer = f"{cite}, se establece:\n\n{content}"

        for q in _question_variants_for_article(num, art_title, de_art + " " + doc_label_short):
            pairs.append(QAPair(question=q, answer=answer, source=doc_name))

    # ── 4. Parágrafos (máx. 4) ────────────────────────────────────────────────
    paragrafos = _extract_paragrafos(text)
    for num, content in paragrafos[:4]:
        q1 = f"¿Qué indica el parágrafo {num} {de_art} {doc_label_short}?"
        q2 = f"¿Qué aclara el parágrafo {num} sobre {doc_label_short} en Unillanos?"
        ans = (
            f"El parágrafo {num} {de_art} {doc_label_short} de Unillanos "
            f"establece:\n\n{content}"
        )
        pairs.append(QAPair(question=q1, answer=ans, source=doc_name))
        pairs.append(QAPair(question=q2, answer=ans, source=doc_name))

    # ── 5. Vigencia ────────────────────────────────────────────────────────────
    vig_m = re.search(
        r"(rige\s+a\s+partir[^.]{10,200}|vigencia[^.]{10,200}|"
        r"a\s+partir\s+de[^.]{10,200})",
        text, re.IGNORECASE,
    )
    if vig_m:
        pairs.append(QAPair(
            question=f"¿Desde cuándo rige {art_lo} {doc_label_short}?",
            answer=(
                f"{intro}, la vigencia indica:\n\n"
                f"{vig_m.group(0).strip()}"
            ),
            source=doc_name,
        ))
        pairs.append(QAPair(
            question=f"¿Cuándo entró en vigor {art_lo} {doc_label_short} de Unillanos?",
            answer=(
                f"{intro}, la vigencia indica:\n\n"
                f"{vig_m.group(0).strip()}"
            ),
            source=doc_name,
        ))

    return pairs


def generate_template(doc_name: str, n: int = 5) -> list[QAPair]:
    """Genera plantillas vacías para completar manualmente."""
    if should_skip(doc_name):
        return []
    doc_type = _detect_doc_type(doc_name)
    ref = _extract_ref(doc_name)
    return [
        QAPair(
            question=f"[PREGUNTA {i+1} sobre {doc_type} {ref}]",
            answer=f"[RESPUESTA {i+1} — completar con información de {doc_name}]",
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
