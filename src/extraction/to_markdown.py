"""Arma el .md estructurado a partir del texto extraido.

Combina dos fuentes: la metadata de la ruta SIRIUS (tipo/año/numero -> frontmatter
YAML) y la estructura legal del cuerpo (CONSIDERANDO/RESUELVE/ARTICULO/PARAGRAFO),
que se detecta con regex y se promueve a encabezados (##, ###, ####) para que
qa_builder pueda recorrerlo por secciones.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


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
    type_label: str        # legible, ej. "Resolucion Superior"
    year: str              # ej. "2023"
    number: str            # ej. "7"
    source_filename: str   # ej. "7_2023.pdf"
    rel_path: str          # ruta relativa al corpus, con / como separador

    @property
    def titulo(self) -> str:
        """Titulo canonico legible: 'Resolucion Superior No. 7 de 2023'."""
        if self.number and self.year:
            return f"{self.type_label} No. {self.number} de {self.year}"
        if self.year:
            return f"{self.type_label} ({self.year})"
        return self.type_label


def parse_meta_from_path(pdf_path: Path, corpus_root: Path) -> DocMeta:
    """Saca tipo, año y numero de la ruta <año>/si/normatividad/<TIPO>/<n>_<año>.pdf; si no calza, rellena lo que pueda."""
    pdf_path = Path(pdf_path)
    try:
        rel = pdf_path.resolve().relative_to(corpus_root.resolve())
    except ValueError:
        rel = Path(pdf_path.name)
    parts = list(rel.parts)

    # si no hay carpeta despues de "normatividad", el tipo es la carpeta padre
    doc_type = ""
    if "normatividad" in parts:
        idx = parts.index("normatividad")
        if idx + 1 < len(parts) - 1:   # debe haber al menos TIPO y archivo despues
            doc_type = parts[idx + 1]
    if not doc_type and len(parts) >= 2:
        doc_type = parts[-2]

    year = ""
    for part in parts:
        if re.fullmatch(r"(19|20)\d{2}", part):
            year = part
            break

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


# pymupdf4llm decora las lineas y a veces vuelve encabezado el ruido: se quita la
# decoracion y solo la estructura legal vuelve a ser encabezado. Las tablas quedan.

# Encabezados de seccion que aparecen solos en una linea.
_SECTION_RE = re.compile(
    r"^(CONSIDERANDO|RESUELVE|R\s*E\s*S\s*U\s*E\s*L\s*V\s*E|ACUERDA|DECRETA|RESOLVIÓ|RESOLVIO|"
    r"DISPONE|ORDENA|CERTIFICA|HACE\s+CONSTAR|EN\s+MÉRITO\s+DE\s+LO\s+EXPUESTO)\s*:?\s*$",
    re.IGNORECASE,
)

# "ARTICULO 8°", "ART. 12", y del OCR "ARTICULO2°." y "ARTÍCULO. 65°". Pegado solo
# si sigue un digito, para no tomar "ARTICULOS" como un articulo "S".
_ARTICULO_RE = re.compile(
    r"^(ART[IÍ]CULO|ART\.)(?:\.?\s+|(?=\d))([A-Za-zÁÉÍÓÚÑ0-9°º]+)\s*[.:\-–)]*\s*(.*)$",
    re.IGNORECASE,
)

# Caracteres chinos, japoneses y coreanos: el OCR los saca de sellos y firmas. En la
# normativa nunca hay texto legitimo con ellos, asi que se borran.
_CJK_RE = re.compile(r"[　-〿㐀-䶿一-鿿豈-﫿]")

# "PARAGRAFO", "PARAGRAFO PRIMERO", "PARAGRAFO 2"
_PARAGRAFO_RE = re.compile(
    r"^(PAR[AÁ]GRAFO)(\s+[A-Za-zÁÉÍÓÚÑ0-9°º]+)?\s*[.:\-–)]*\s*(.*)$",
    re.IGNORECASE,
)

# Ruido de pie/encabezado de pagina: "Pagina 1 de 2", "Pag. 3"
_PAGE_NOISE_RE = re.compile(r"^P[áa]g(?:ina|\.)?\s*\d+\s*(?:de\s*\d+)?$", re.IGNORECASE)

# Banner repetido en cada hoja. Anclado al inicio de linea para no borrar un
# articulo que mencione "Universidad de los Llanos".
_BANNER_RE = re.compile(
    r"^UNIVERSIDAD\s+DE\s+LOS\s+LLANOS\b.*(RESOLUCI[ÓO]N|ACUERDO|DECRETO|CONSEJO)"
    r"|^(RESOLUCI[ÓO]N|ACUERDO|DECRETO)\s.{0,75}\bNo?[°º.]?\s*\d+.{0,20}\bDE\s+(19|20)\d{2}\b\)?\.?\s*(\(.*\))?$",
    re.IGNORECASE,
)

# Membrete de la hoja siguiente que el OCR deja a mitad de un articulo. Solo si es la
# linea COMPLETA (y en mayusculas): "Consejo Academico" dentro de una frase no se toca.
_MEMBRETE_RE = re.compile(
    r"^UNIVERSIDAD\s+DE\s+LOS\s+LLA[NÑ]OS\.?$"
    r"|^CONSEJO\s+(ACAD[EÉ]MICO|SUPERIOR(\s+UNIVERSITARIO)?)\.?$"
    r"|^CONSEJO\s+(ACAD[EÉ]MICO|SUPERIOR(\s+UNIVERSITARIO)?)\s+(RESOLUCI[ÓO]N|ACUERDO)\b.{0,60}$"
    r"|^(RESOLUCI[ÓO]N|ACUERDO)\s+(RECTORAL|ACAD[EÉ]MICA?O?|SUPERIOR)\s+No?[°º.]?$",
    re.IGNORECASE,
)
# El mismo encabezado con las letras mal espaciadas por el OCR: "ACUERD OACA DÉM IC ON°022DE2015
# (Septiembre 29)". Se compara la linea SIN espacios.
_MEMBRETE_PEGADO_RE = re.compile(
    r"^(ACUERDO|RESOLUCI[ÓO]N)(ACAD[EÉ]MIC[OA]|SUPERIOR|RECTORAL)?N[°º.]?\d{1,4}DE(19|20)\d{2}(\(.{0,25}\))?\.?$",
    re.IGNORECASE,
)

# Marca de pagina del OCR: "Hoja N°. 2", "Hoja 3".
_HOJA_RE = re.compile(r"^Hoja\s*\S{0,4}\s*\d{1,3}$", re.IGNORECASE)

# Linea de fecha suelta tipo "( Mayo 05 )".
_DATE_LINE_RE = re.compile(r"^\(\s*[A-Za-zÁÉÍÓÚáéíóú]+\.?\s+\d{1,2}\s*\)$")

# Separador de pagina / regla horizontal que inserta pymupdf4llm.
_HR_RE = re.compile(r"^-{3,}$")

# Pie de pagina institucional de Unillanos (direccion, conmutador, correo suelto).
_FOOTER_RE = re.compile(
    r"(Kil[óo]metro\s+\d+\s+V[íi]a\s+a\s+Puerto\s+L[óo]pez|Vereda\s+Barcelona|"
    r"Conmutador\s*\d|^[\w.+-]+@[\w.-]+\.\w+$)",
    re.IGNORECASE,
)

# Formulas de cierre: desde aqui empiezan la fecha y las firmas, ya no es texto del articulo.
# Tolera lo que deja el OCR: tildes cambiadas ("PÚBLIQUESE") y espacios de mas o de menos.
_CIERRE_RE = re.compile(
    r"P[UÚ]BL[IÍ]\s*QUESE|COM[UÚ]N[IÍ]\s*QUESE|C[UÚ]\s*MPLASE|NOTIF[IÍ]\s*QUESE|"
    r"DAD[AO]\s+EN\s+VILLAVICENCIO|PROYECT[OÓ]\s*:",
    re.IGNORECASE,
)

# "CAPITULO I", "TITULO II", "CAPITULO PRIMERO" — divisiones del articulado.
# El ordinal se restringe a romano/ordinal/digito para no capturar "Titulo de..." del cuerpo.
_ORDINAL = (r"(?:[IVXLC]+|\d+|PRIMER[OA]?|SEGUND[OA]|TERCER[OA]?|CUART[OA]|QUINT[OA]|"
            r"SEXT[OA]|S[EÉ]PTIM[OA]|OCTAV[OA]|NOVEN[OA]|D[EÉ]CIM[OA])")
_CAPITULO_RE = re.compile(
    rf"^(CAP[IÍ]TULO|T[IÍ]TULO)\s+({_ORDINAL})\b\s*[.:\-–)]*\s*(.*)$",
    re.IGNORECASE,
)


def _strip_decoration(line: str) -> str:
    """Quita la decoracion Markdown de pymupdf4llm para dejar texto plano."""
    s = line.strip()
    s = re.sub(r"<!--.*?-->", "", s)           # comentarios (<!-- image --> de docling)
    s = re.sub(r"^#{1,6}\s*", "", s)          # encabezados ATX
    s = re.sub(r"</?[a-zA-Z][^>]*>", "", s)    # etiquetas HTML (<mark>, <u>, <br>...)
    s = re.sub(r"[*`]+", "", s)                # negrita/italica/codigo
    s = re.sub(r"_+", " ", s)                  # italicas con guion bajo
    s = s.replace("\\", " ")                   # backslashes sueltos de pymupdf
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def _separar_cierre(lines: list[str]) -> list[str]:
    """Pone "## CIERRE" donde empiezan las firmas, para que no queden dentro del ultimo articulo.
    Solo mira despues del ultimo articulo (hay anexos despues de las firmas) y no corta si
    el articulo quedaria casi vacio (uno que empiece "Comuniquese la...")."""
    arts = [i for i, l in enumerate(lines) if l.startswith("### ")]
    if not arts:
        return lines
    texto = 0   # caracteres del ultimo articulo antes del posible corte
    for i in range(arts[-1] + 1, len(lines)):
        l = lines[i]
        if l.startswith("#"):          # paragrafos del mismo articulo
            continue
        m = None if l.startswith("|") else _CIERRE_RE.search(l)
        if m and texto + m.start() >= 20:
            antes, despues = l[:m.start()].rstrip(), l[m.start():]
            return lines[:i] + ([antes] if antes else []) + ["", "## CIERRE", "", despues] + lines[i + 1:]
        texto += len(l)
    return lines


def _normalize_markdown(md: str) -> str:
    """Quita el ruido de pagina, deja una vez las lineas repetidas y vuelve encabezados
    las secciones (##), articulos (###) y paragrafos (####). Las tablas quedan igual."""
    raw_lines = [_CJK_RE.sub("", l) for l in md.splitlines()]

    # Frecuencia de lineas de texto (sin decoracion) para deduplicar repetidos.
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

        if (_PAGE_NOISE_RE.match(s) or _DATE_LINE_RE.match(s)
                or _BANNER_RE.search(s) or _FOOTER_RE.search(s) or _HOJA_RE.match(s)
                or (s[:10].isupper() and _MEMBRETE_RE.match(s))
                or _MEMBRETE_PEGADO_RE.match(s.replace(" ", ""))):
            continue

        # Texto corrido repetido (subtitulo por hoja): conservar 1a aparicion.
        key = s.lower()
        if freq.get(key, 0) >= 2:
            if key in seen_repeat:
                continue
            seen_repeat.add(key)

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

    out_lines = _separar_cierre(out_lines)
    md_out = "\n".join(out_lines)
    md_out = re.sub(r"\n{3,}", "\n\n", md_out)
    return md_out.strip()


# Encabezado impreso: "ACUERDO SUPERIOR No. 016 DE 2020". Se busca sin espacios porque
# el OCR escribe "A C U E R D O  S U P E R I O R".
_IMPRESO_RE = re.compile(
    r"^(?:UNIVERSIDADDELOSLLA[NÑ]OS)?(?:CONSEJO(?:ACAD[EÉ]MICO|SUPERIOR(?:UNIVERSITARIO)?))?"
    r"(ACUERDO|RESOLUCI[OÓ]N|DECRETO|LEY)[A-ZÁÉÍÓÚÑ\-]{0,40}?N(?:O|Ú?MERO)?[°º.:]*0*(\d{1,4})DE((?:19|20)\d{2})"
)


def _numero_impreso(text: str) -> tuple[str, str]:
    """(numero, año) del encabezado impreso, leido antes de la limpieza que lo borra.
    Detecta PDF mal nombrados: el archivo 49_2015.pdf es la RS 050 de 2015."""
    for linea in text.splitlines()[:40]:
        s = _strip_decoration(linea).upper().replace(" ", "")
        m = _IMPRESO_RE.match(s)
        if m:
            return m.group(2), m.group(3)
    return "", ""


def _yaml_frontmatter(meta: DocMeta, method: str, pages: int, char_count: int,
                      impreso: tuple[str, str] = ("", "")) -> str:
    def esc(v: str) -> str:
        return '"' + str(v).replace('"', '\\"') + '"'
    return (
        "---\n"
        f"tipo: {meta.doc_type}\n"
        f"tipo_legible: {esc(meta.type_label)}\n"
        f"numero: {esc(meta.number)}\n"
        f"anio: {esc(meta.year)}\n"
        f"numero_impreso: {esc(impreso[0])}\n"
        f"anio_impreso: {esc(impreso[1])}\n"
        f"titulo: {esc(meta.titulo)}\n"
        f"fuente: {esc(meta.source_filename)}\n"
        f"ruta_origen: {esc(meta.rel_path)}\n"
        f"metodo_extraccion: {method}\n"
        f"paginas: {pages}\n"
        f"caracteres: {char_count}\n"
        "---\n"
    )


def to_markdown(text: str, meta: DocMeta, method: str, pages: int) -> str:
    """Frontmatter + titulo + cuerpo normalizado, a partir del texto crudo extraido."""
    body = _normalize_markdown(text)
    frontmatter = _yaml_frontmatter(meta, method, pages, len(text), _numero_impreso(text))
    return f"{frontmatter}\n# {meta.titulo}\n\n{body}\n"
