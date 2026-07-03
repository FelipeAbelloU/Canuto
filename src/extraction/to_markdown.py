"""Arma el .md estructurado a partir del texto extraído.

Combina dos fuentes: la metadata de la ruta SIRIUS (tipo/año/número -> frontmatter
YAML) y la estructura legal del cuerpo (CONSIDERANDO/RESUELVE/ARTÍCULO/PARÁGRAFO),
que se detecta con regex y se promueve a encabezados (##, ###, ####) para que
qa_builder pueda recorrerlo por secciones.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# --- Etiquetas legibles por tipo de carpeta -------------------------------
TYPE_LABELS = {
    "ACTA_CONSEJO_ACADEMICO": "Acta del Consejo Académico",
    "ACTA_CONSEJO_SUPERIOR": "Acta del Consejo Superior",
    "ACUERDO_ACADEMICO": "Acuerdo Académico",
    "ACUERDO_CESU": "Acuerdo CESU",
    "ACUERDO_SUPERIOR": "Acuerdo Superior",
    "CONPES": "Documento CONPES",
    "Contenido_Programatico": "Contenido Programático",
    "DECRETO_MEN": "Decreto del MEN",
    "DECRETO_NACIONAL": "Decreto Nacional",
    "LEY": "Ley",
    "MALLA_CURRICULAR": "Malla Curricular",
    "Malla_curricular": "Malla Curricular",
    "ORDENANZA_ASAMBLEA_DEL_DEPARTAMENTO_DEL_META": "Ordenanza de la Asamblea del Meta",
    "Plan": "Plan",
    "PLAN_DE_ESTUDIOS_DEL_PROGRAMA": "Plan de Estudios del Programa",
    "Plan_de_estudios_del_programa": "Plan de Estudios del Programa",
    "RESOLUCION_ACADEMICA": "Resolución Académica",
    "RESOLUCION_CONSEJO_DE_FACULTAD_-_FCBI": "Resolución del Consejo de Facultad (FCBI)",
    "RESOLUCION_MEN": "Resolución del MEN",
    "RESOLUCION_MINISTERIO_DE_AMBIENTE_NACIONAL": "Resolución del Ministerio de Ambiente",
    "RESOLUCION_MINISTERIO_DE_EDUCACION_NACIONAL": "Resolución del Ministerio de Educación Nacional",
    "Resolucion_Ministerio_de_Educacion_Nacional": "Resolución del Ministerio de Educación Nacional",
    "RESOLUCION_RECTORAL": "Resolución Rectoral",
    "RESOLUCION_SUPERIOR": "Resolución Superior",
}


@dataclass
class DocMeta:
    """Metadata derivada de la ruta del PDF dentro del corpus."""
    doc_type: str          # nombre de la carpeta, ej. "RESOLUCION_SUPERIOR"
    type_label: str        # legible, ej. "Resolución Superior"
    year: str              # ej. "2023"
    number: str            # ej. "7"
    source_filename: str   # ej. "7_2023.pdf"
    rel_path: str          # ruta relativa al corpus, con / como separador

    @property
    def titulo(self) -> str:
        """Título canónico legible: 'Resolución Superior No. 7 de 2023'."""
        if self.number and self.year:
            return f"{self.type_label} No. {self.number} de {self.year}"
        if self.year:
            return f"{self.type_label} ({self.year})"
        return self.type_label


def parse_meta_from_path(pdf_path: Path, corpus_root: Path) -> DocMeta:
    """Deriva DocMeta de la ruta ``<año>/si/normatividad/<TIPO>/<n>_<año>.pdf``.

    Es tolerante: si la estructura no calza exactamente, rellena lo que pueda.
    """
    pdf_path = Path(pdf_path)
    try:
        rel = pdf_path.resolve().relative_to(corpus_root.resolve())
    except ValueError:
        rel = Path(pdf_path.name)
    parts = list(rel.parts)

    # Tipo: la carpeta que sigue a "normatividad"; si no existe, la carpeta padre.
    doc_type = ""
    if "normatividad" in parts:
        idx = parts.index("normatividad")
        if idx + 1 < len(parts) - 1:   # debe haber al menos TIPO y archivo después
            doc_type = parts[idx + 1]
    if not doc_type and len(parts) >= 2:
        doc_type = parts[-2]

    # Año: la primera parte de 4 dígitos en la ruta.
    year = ""
    for part in parts:
        if re.fullmatch(r"(19|20)\d{2}", part):
            year = part
            break

    # Número: lo que precede al primer "_" en el nombre, si es numérico.
    stem = pdf_path.stem
    number = ""
    m = re.match(r"(\d+)[_\-]", stem)
    if m:
        number = m.group(1)
    elif stem.isdigit():
        number = stem

    type_label = TYPE_LABELS.get(doc_type) or doc_type.replace("_", " ").title()

    return DocMeta(
        doc_type=doc_type,
        type_label=type_label,
        year=year,
        number=number,
        source_filename=pdf_path.name,
        rel_path=rel.as_posix(),
    )


# --- Detección de estructura legal ----------------------------------------
#
# La entrada es el Markdown de pymupdf4llm, que trae las líneas DECORADAS
# (`## **CONSIDERANDO**`, `**Artículo 1º.-OBJETO.**`) y a veces promueve ruido a
# encabezados. Por eso NO se confía en sus encabezados: se quita la decoración de
# cada línea y se re-promueve SOLO la estructura legal con estas regex. Las tablas
# se preservan tal cual (es la ventaja de pymupdf4llm frente a pdfplumber).

# Encabezados de sección que aparecen solos en una línea.
_SECTION_RE = re.compile(
    r"^(CONSIDERANDO|RESUELVE|R\s*E\s*S\s*U\s*E\s*L\s*V\s*E|ACUERDA|DECRETA|RESOLVIÓ|RESOLVIO|"
    r"DISPONE|ORDENA|CERTIFICA|HACE\s+CONSTAR|EN\s+MÉRITO\s+DE\s+LO\s+EXPUESTO)\s*:?\s*$",
    re.IGNORECASE,
)

# "ARTÍCULO PRIMERO", "ARTÍCULO 8.", "ARTÍCULO 8°", "ART. 12"
_ARTICULO_RE = re.compile(
    r"^(ART[IÍ]CULO|ART\.)\s+([A-Za-zÁÉÍÓÚÑ0-9°º]+)\s*[.:\-–)]*\s*(.*)$",
    re.IGNORECASE,
)

# "PARÁGRAFO", "PARÁGRAFO PRIMERO", "PARÁGRAFO 2"
_PARAGRAFO_RE = re.compile(
    r"^(PAR[AÁ]GRAFO)(\s+[A-Za-zÁÉÍÓÚÑ0-9°º]+)?\s*[.:\-–)]*\s*(.*)$",
    re.IGNORECASE,
)

# Ruido de pie/encabezado de página: "Página 1 de 2", "Pág. 3"
_PAGE_NOISE_RE = re.compile(r"^P[áa]g(?:ina|\.)?\s*\d+\s*(?:de\s*\d+)?$", re.IGNORECASE)

# Banner institucional repetido en cada hoja (se descarta: el título canónico ya
# viene del frontmatter/metadata). Anclado al INICIO de línea para no borrar texto
# de cuerpo que mencione "Universidad de los Llanos" (p.ej. dentro de un artículo).
_BANNER_RE = re.compile(
    r"^UNIVERSIDAD\s+DE\s+LOS\s+LLANOS\b.*(RESOLUCI[ÓO]N|ACUERDO|DECRETO|CONSEJO)"
    r"|^(RESOLUCI[ÓO]N|ACUERDO|DECRETO)\s.{0,75}\bNo?[°º.]?\s*\d+.{0,20}\bDE\s+(19|20)\d{2}\b\)?\.?\s*(\(.*\))?$",
    re.IGNORECASE,
)

# Línea de fecha suelta tipo "( Mayo 05 )".
_DATE_LINE_RE = re.compile(r"^\(\s*[A-Za-zÁÉÍÓÚáéíóú]+\.?\s+\d{1,2}\s*\)$")

# Separador de página / regla horizontal que inserta pymupdf4llm.
_HR_RE = re.compile(r"^-{3,}$")

# Pie de página institucional de Unillanos (dirección, conmutador, correo suelto).
_FOOTER_RE = re.compile(
    r"(Kil[óo]metro\s+\d+\s+V[íi]a\s+a\s+Puerto\s+L[óo]pez|Vereda\s+Barcelona|"
    r"Conmutador\s*\d|^[\w.+-]+@[\w.-]+\.\w+$)",
    re.IGNORECASE,
)

# "CAPÍTULO I", "TÍTULO II", "CAPITULO PRIMERO" — divisiones del articulado.
# El ordinal se restringe a romano/ordinal/dígito para no capturar "Título de..." del cuerpo.
_ORDINAL = (r"(?:[IVXLC]+|\d+|PRIMER[OA]?|SEGUND[OA]|TERCER[OA]?|CUART[OA]|QUINT[OA]|"
            r"SEXT[OA]|S[EÉ]PTIM[OA]|OCTAV[OA]|NOVEN[OA]|D[EÉ]CIM[OA])")
_CAPITULO_RE = re.compile(
    rf"^(CAP[IÍ]TULO|T[IÍ]TULO)\s+({_ORDINAL})\b\s*[.:\-–)]*\s*(.*)$",
    re.IGNORECASE,
)


def _strip_decoration(line: str) -> str:
    """Quita la decoración Markdown de pymupdf4llm para dejar texto plano."""
    s = line.strip()
    s = re.sub(r"<!--.*?-->", "", s)           # comentarios (<!-- image --> de docling)
    s = re.sub(r"^#{1,6}\s*", "", s)          # encabezados ATX
    s = re.sub(r"</?[a-zA-Z][^>]*>", "", s)    # etiquetas HTML (<mark>, <u>, <br>...)
    s = re.sub(r"[*`]+", "", s)                # negrita/itálica/código
    s = re.sub(r"_+", " ", s)                  # itálicas con guion bajo
    s = s.replace("\\", " ")                   # backslashes sueltos de pymupdf
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def _normalize_markdown(md: str) -> str:
    """Normaliza el Markdown de pymupdf4llm a estructura legal limpia.

    - Preserva bloques de tabla tal cual.
    - Descarta encabezados-ruido, banners de página, fechas sueltas y reglas.
    - Deduplica líneas de texto corrido repetidas (subtítulos que se repiten por hoja).
    - Re-promueve CONSIDERANDO/RESUELVE→##, ARTÍCULO→###, PARÁGRAFO→####.
    """
    raw_lines = md.splitlines()

    # Frecuencia de líneas de texto (sin decoración) para deduplicar repetidos.
    from collections import Counter
    freq: Counter[str] = Counter()
    for raw in raw_lines:
        if _is_table_line(raw):
            continue
        s = _strip_decoration(raw)
        if len(s) >= 25:
            freq[s.lower()] += 1

    out_lines: list[str] = []
    seen_repeat: set[str] = set()

    for raw in raw_lines:
        # Tablas: preservar verbatim.
        if _is_table_line(raw):
            out_lines.append(raw.rstrip())
            continue

        line = raw.rstrip()
        if not line.strip():
            out_lines.append("")
            continue
        if _HR_RE.match(line.strip()):
            continue

        s = _strip_decoration(line)
        if not s:
            out_lines.append("")
            continue

        # Ruido a descartar siempre.
        if (_PAGE_NOISE_RE.match(s) or _DATE_LINE_RE.match(s)
                or _BANNER_RE.search(s) or _FOOTER_RE.search(s)):
            continue

        # Texto corrido repetido (subtítulo por hoja): conservar 1a aparición.
        key = s.lower()
        if freq.get(key, 0) >= 2:
            if key in seen_repeat:
                continue
            seen_repeat.add(key)

        # Promoción de estructura legal.
        m_sec = _SECTION_RE.match(s)
        if m_sec:
            label = re.sub(r"\s+", "", m_sec.group(1)).upper()
            out_lines += ["", f"## {label}", ""]
            continue

        m_cap = _CAPITULO_RE.match(s)
        if m_cap:
            label = f"{m_cap.group(1).upper()} {m_cap.group(2).upper()}"
            resto = m_cap.group(3).strip()
            out_lines += ["", f"## {label}" + (f" — {resto}" if resto else "")]
            continue

        m_art = _ARTICULO_RE.match(s)
        if m_art:
            label = f"ARTÍCULO {m_art.group(2).upper()}"
            resto = m_art.group(3).strip()
            out_lines += ["", f"### {label}"]
            if resto:
                out_lines.append(resto)
            continue

        m_par = _PARAGRAFO_RE.match(s)
        if m_par:
            ordinal = (m_par.group(2) or "").strip()
            label = "PARÁGRAFO" + (f" {ordinal.upper()}" if ordinal else "")
            resto = m_par.group(3).strip()
            out_lines += ["", f"#### {label}"]
            if resto:
                out_lines.append(resto)
            continue

        out_lines.append(s)

    md_out = "\n".join(out_lines)
    md_out = re.sub(r"\n{3,}", "\n\n", md_out)
    return md_out.strip()


def _yaml_frontmatter(meta: DocMeta, method: str, pages: int, char_count: int) -> str:
    def esc(v: str) -> str:
        return '"' + str(v).replace('"', '\\"') + '"'
    return (
        "---\n"
        f"tipo: {meta.doc_type}\n"
        f"tipo_legible: {esc(meta.type_label)}\n"
        f"numero: {esc(meta.number)}\n"
        f"anio: {esc(meta.year)}\n"
        f"titulo: {esc(meta.titulo)}\n"
        f"fuente: {esc(meta.source_filename)}\n"
        f"ruta_origen: {esc(meta.rel_path)}\n"
        f"metodo_extraccion: {method}\n"
        f"paginas: {pages}\n"
        f"caracteres: {char_count}\n"
        "---\n"
    )


def to_markdown(text: str, meta: DocMeta, method: str, pages: int) -> str:
    """Construye el documento Markdown completo: frontmatter + título + cuerpo.

    ``text`` es el Markdown crudo de pymupdf4llm (o texto plano de un fallback);
    se normaliza a estructura legal limpia antes de escribirlo.
    """
    body = _normalize_markdown(text)
    frontmatter = _yaml_frontmatter(meta, method, pages, len(text))
    return f"{frontmatter}\n# {meta.titulo}\n\n{body}\n"
