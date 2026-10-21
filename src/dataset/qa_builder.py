"""Genera los pares pregunta-respuesta a partir de los documentos en Markdown.

Lee el .md que produce extract_text.py y usa el frontmatter (tipo, numero, año,
titulo) para armar la cita, y los encabezados (## CONSIDERANDO/RESUELVE,
### ARTICULO N, #### PARAGRAFO) para sacar el contenido de cada articulo.

Formato de salida (alpaca): {"instruction": ..., "input": ..., "output": ...}
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


def to_alpaca(pair: QAPair) -> dict:
    return {"instruction": pair.question, "input": "", "output": pair.answer}


# ---------------------------------------------------------------------------
# Tipos de documento que NO generan buenos QA (tablas, mallas, planes de estudio)
# ---------------------------------------------------------------------------
_SKIP_TYPES = {
    "MALLA_CURRICULAR", "Malla_curricular",
    "PLAN_DE_ESTUDIOS_DEL_PROGRAMA", "Plan_de_estudios_del_programa",
    "Contenido_Programatico", "Plan",
}


def _skip_type(tipo: str) -> bool:
    # Estos tipos son tablas: el texto plano no produce preguntas utiles.
    return tipo in _SKIP_TYPES


# ---------------------------------------------------------------------------
# Concordancia gramatical segun el tipo (la Resolucion / el Acuerdo)
# ---------------------------------------------------------------------------
def _gender(tipo_legible: str) -> tuple[str, str, str]:
    """Devuelve (articulo, articulo_mayuscula, contraccion) para el tipo.

    'la Resolucion / La / de la'  ·  'el Acuerdo / El / del'
    """
    t = tipo_legible.lower()
    # Acta es femenino pero lleva articulo masculino ("el acta", "del acta").
    if t.startswith(("acuerdo", "decreto", "acta", "documento", "reglamento", "plan")):
        return ("el", "El", "del")
    return ("la", "La", "de la")


# Tipos que son legislacion NACIONAL, no normativa propia de Unillanos:
# se citan sin "de Unillanos" para no atribuirle autoria equivocada.
_NATIONAL_HINTS = ("NACIONAL", "MINISTERIO", "_MEN", "CONPES", "LEY", "ORDENANZA", "CESU")


def _is_national(tipo: str) -> bool:
    u = tipo.upper()
    return u == "LEY" or any(h in u for h in _NATIONAL_HINTS)


# Un numero de articulo valido: arabigo, romano u ordinal escrito.
_ROMAN_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
_ORDINAL_WORDS = {
    "primero", "segundo", "tercero", "cuarto", "quinto", "sexto", "séptimo",
    "septimo", "octavo", "noveno", "décimo", "decimo", "único", "unico",
}


def _valid_num(n: str) -> str:
    """Devuelve el numero si es valido (arabigo/romano/ordinal), o '' si no."""
    n = n.strip()
    if re.fullmatch(r"\d{1,3}", n):
        return n
    # Numeracion con puntos de los decretos nacionales: "2.5.3.2.1.1"
    if re.fullmatch(r"\d+(\.\d+)+", n):
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

# Secciones despues de las cuales empiezan los articulos propios del documento.
_RESOL = {"RESUELVE", "ACUERDA", "DECRETA", "DISPONE", "ORDENA", "RESOLVIÓ", "RESOLVIO"}
# El resto de la numeracion con puntos queda al inicio del texto: "5.3.2.1.1. Titulo..."
_PUNTOS_RE = re.compile(r"^(\d+(?:\.\d+)+)[.\s]+")
# Cuantos encabezados se mira hacia adelante para ver si la numeracion se retoma.
# Con 2, 3, 4, 6 o 10 da lo mismo; sin limite se traga articulos propios (Ley 1955).
_VENTANA = 4
_ORDINAL_VALOR = {
    "primero": 1, "segundo": 2, "tercero": 3, "cuarto": 4, "quinto": 5, "sexto": 6,
    "séptimo": 7, "septimo": 7, "octavo": 8, "noveno": 9, "décimo": 10, "decimo": 10,
    "único": 1, "unico": 1,
}


@dataclass
class DocStructure:
    descripcion: str = ""                       # el "Por la cual…" del encabezado
    considerando: str = ""                      # texto bajo ## CONSIDERANDO
    resuelve: str = ""                           # texto bajo ## RESUELVE antes del 1er articulo
    articles: list[tuple[str, str]] = field(default_factory=list)      # (numero, contenido)
    # Para el reporte (no generan pares):
    citados_considerando: int = 0               # "articulos" de otra norma citados en el considerando
    pegados: int = 0                             # articulos de otra norma pegados al que los modifica
    fuera: list[str] = field(default_factory=list)    # numeros que la red de seguridad deja por fuera
    huecos: list[int] = field(default_factory=list)   # numeros que faltan en la secuencia 1, 2, 3...


def _join(lines: list[str]) -> str:
    # Une las lineas de un bloque en un parrafo limpio.
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _clean_art_num(raw: str) -> str:
    # Limpia el numero del encabezado "### ARTICULO 5°." -> "5".
    num = raw.strip().strip(".:°º-–) ").strip()
    # "1O" es "1º" mal leido por el OCR (letra O en vez del simbolo).
    m = re.fullmatch(r"(\d{1,3})[Oo]", num)
    return m.group(1) if m else num


def _num_valor(num: str) -> int | None:
    # Valor entero del numero para seguir la secuencia (None si no se puede).
    if num.isdigit():
        return int(num)
    return _ORDINAL_VALOR.get(num.lower())


def _ordenar_articulos(articles: list[list]) -> tuple[list[tuple[str, str]], int]:
    """Regla de secuencia: separa los articulos propios del texto citado de otra norma.

    Los articulos propios van en orden (1, 2, 3...). Si aparece un numero fuera de orden
    y la cuenta se retoma dentro de los _VENTANA encabezados siguientes, esos encabezados
    son texto citado (una norma que modifica otra) y se pegan al articulo anterior.
    Ej. RS 30/2021: 1 2 7 3 8 4 9 ... -> 1 2 3 4 ... (el 7, 8, 9 son de la RS 20/2021).
    Si no se retoma, es un hueco (un encabezado perdido) y se acepta como propio.
    """
    # Numeracion con puntos: "ARTICULO 2" + "5.3.2.1.1. Titulo" -> "2.5.3.2.1.1"
    cand = []
    for num, txt in articles:
        mp = _PUNTOS_RE.match(txt) if num.isdigit() else None
        if mp:
            cand.append((num + "." + mp.group(1), txt[mp.end():], None))
        else:
            cand.append((num, txt, _num_valor(num) if _valid_num(num) else None))

    propios: list[list] = []
    pegados = 0
    esperado = None
    i = 0
    while i < len(cand):
        num, txt, v = cand[i]
        if v is None or esperado is None or v == esperado:
            propios.append([num, txt])
            if v is not None:
                esperado = v + 1
            i += 1
            continue
        # Fuera de orden: buscar si la numeracion se retoma pronto.
        j = None
        for k in range(i + 1, min(i + 1 + _VENTANA, len(cand))):
            if cand[k][2] == esperado:
                j = k
                break
        if j is not None and propios:
            for k in range(i, j):
                propios[-1][1] += f" Artículo {cand[k][0]}. {cand[k][1]}"
                pegados += 1
            i = j
            continue
        propios.append([num, txt])
        if v > esperado:
            esperado = v + 1
        i += 1
    return [(n, t) for n, t in propios], pegados


def _parse_structure(body_md: str) -> DocStructure:
    """Recorre el Markdown por encabezados y separa encabezado, secciones y articulos."""
    lines = body_md.splitlines()

    header: list[str] = []          # texto antes del primer ##
    sections: dict[str, list[str]] = {}
    articles: list[list] = []       # [num, [lineas]]
    target = header                 # donde se acumulan las lineas de contenido actuales

    # Si el documento tiene RESUELVE/ACUERDA, los "articulos" que aparecen antes son
    # citas de otras normas dentro del considerando, no articulos propios.
    tiene_resol = any(
        _H2_RE.match(l) and not l.startswith("###")
        and _H2_RE.match(l).group(1).strip().upper() in _RESOL
        for l in lines
    )
    visto_resol = False
    citados = 0

    for raw in lines:
        s = raw.rstrip()

        # Titulo del documento (# ...): se ignora, ya esta en el frontmatter.
        if s.startswith("# ") and not s.startswith("## "):
            continue

        # Articulo (### ARTICULO N) -> nuevo bloque de articulo.
        m_art = _ART_RE.match(s)
        if m_art:
            num = _clean_art_num(m_art.group(1))
            if tiene_resol and not visto_resol:
                # Cita dentro del considerando: su texto sigue siendo considerando.
                target.append(f"Artículo {num}.")
                citados += 1
                continue
            articles.append([num, []])
            target = articles[-1][1]
            continue

        # Paragrafo (#### PARAGRAFO): pertenece al articulo actual, asi que su
        # contenido se pliega dentro del bloque vigente con una etiqueta.
        m_par = _PAR_RE.match(s)
        if m_par:
            num_par = m_par.group(1).strip()
            target.append(f"Parágrafo{(' ' + num_par) if num_par else ''}:")
            continue

        # Seccion (## CONSIDERANDO / RESUELVE / CAPITULO ...).
        m_h2 = _H2_RE.match(s)
        if m_h2 and not s.startswith("### "):
            name = m_h2.group(1).strip().upper()
            if name in _RESOL:
                visto_resol = True
            sections.setdefault(name, [])
            target = sections[name]
            continue

        # Linea de contenido normal.
        target.append(s)

    st = DocStructure()
    st.descripcion = _extract_descripcion(header)
    st.considerando = _join(sections.get("CONSIDERANDO", []))
    st.resuelve = _join(sections.get("RESUELVE", []) or sections.get("ACUERDA", []))
    st.citados_considerando = citados

    propios, st.pegados = _ordenar_articulos([(num, _join(ls)) for num, ls in articles])

    # Red de seguridad: numero invalido, repetido o sin texto -> ese articulo no genera
    # pares. Nunca se inventa un numero con la posicion.
    cuenta: dict[str, int] = {}
    for num, _ in propios:
        cuenta[num] = cuenta.get(num, 0) + 1
    for num, txt in propios:
        if txt and cuenta[num] == 1 and _valid_num(num):
            st.articles.append((num, txt))
        else:
            st.fuera.append(num)

    # Huecos en la numeracion (1, 2, 4 -> falta el 3). Solo se reportan: el articulo
    # anterior a un hueco puede traer pegado el texto del que falta (AS 20/2021, art. 64-65).
    enteros = {int(n) for n, _ in st.articles if n.isdigit()}
    if enteros:
        st.huecos = [n for n in range(1, max(enteros) + 1) if n not in enteros]
    return st


def _extract_descripcion(header_lines: list[str]) -> str:
    """Saca el 'Por la cual…' del encabezado, con o sin comillas."""
    header = " ".join(header_lines)
    m = re.search(r'[“"«]\s*(Por\s[^”"»]{15,400})[”"»]', header, re.IGNORECASE)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()

    # Sin comillas (40 % de los documentos por OCR): desde la linea "Por el/la cual..."
    # hasta la autoridad que firma ("EL CONSEJO ACADEMICO", "EL RECTOR", "En uso de...").
    partes: list[str] = []
    for linea in header_lines:
        s = linea.strip()
        if not partes:
            if re.match(r'^[“"«]?\s*Por\s+(el|la|lo|medio)\b', s, re.IGNORECASE):
                partes.append(s)
            continue
        if not s:
            continue
        if re.match(r"^(E[LI]|LA)\s*[A-ZÁÉÍÓÚÑ]{3,}", s) or re.match(r"^En\s+uso\b", s, re.IGNORECASE):
            break
        partes.append(s)
    desc = re.sub(r"\s+", " ", " ".join(partes)).strip(' “”"«»')
    return desc if 15 <= len(desc) <= 400 else ""


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


def generate_heuristic(body_md: str, meta: dict) -> list[QAPair]:
    """Genera los pares QA fundamentados de un documento en Markdown (sin tope: todos sus articulos).

    Estrategia:
      1. Pregunta(s) general(es) sobre de que trata el documento.
      2. Una pregunta por articulo, con el texto real del articulo como respuesta.
      3. Fallback (documentos sin articulos): usar CONSIDERANDO/RESUELVE.
    Todas las respuestas citan el documento por su numero y año exactos.
    """
    if _skip_type(meta.get("tipo", "")):
        return []

    tipo, cita, _ = _doc_ref(meta)
    la, La, de_la = _gender(tipo)

    # Legislacion nacional: se cita sin atribuirla a Unillanos.
    nacional = _is_national(meta.get("tipo", ""))
    suf_q = "" if nacional else " de Unillanos"                       # sufijo de la pregunta
    entidad = "" if nacional else " de la Universidad de los Llanos (Unillanos)"  # en la respuesta

    st = _parse_structure(body_md)
    pairs: list[QAPair] = []

    # ── 1. Preguntas generales ────────────────────────────────────────────
    if st.descripcion:
        intro_ans = f"{La} {cita}{entidad} es la norma «{st.descripcion}»."
        # Añadir el primer articulo como contexto de lo que dispone.
        if st.articles:
            intro_ans += f" En su artículo {st.articles[0][0] or '1'} establece: {st.articles[0][1]}"
        pairs.append(QAPair(f"¿De qué trata {la} {cita}{suf_q}?", intro_ans))
        pairs.append(QAPair(f"¿Qué regula {la} {cita}{suf_q}?", intro_ans))

    # ── 2. Un par por articulo (el nucleo del dataset) ────────────────────
    # El parser ya garantiza un numero UNICO y valido por articulo (los dudosos
    # quedan fuera), asi que no hay preguntas identicas con respuestas distintas.
    for raw_num, content in st.articles:
        num = _valid_num(raw_num)

        q = f"¿Qué establece el artículo {num} {de_la} {cita}{suf_q}?"
        a = f"Según el artículo {num} {de_la} {cita}{entidad}, se establece: {content}"
        pairs.append(QAPair(q, a))

    # ── 3. Fallback: documentos sin articulos (actas, comunicados) ────────
    if not st.articles:
        cuerpo = st.resuelve or st.considerando or st.descripcion
        if cuerpo:
            if not st.descripcion:
                pairs.append(QAPair(
                    f"¿Qué dispone {la} {cita}{suf_q}?",
                    f"{La} {cita}{entidad} dispone: {cuerpo[:1200]}",
                ))
            if st.considerando:
                pairs.append(QAPair(
                    f"¿Cuál es el fundamento {de_la} {cita}{suf_q}?",
                    f"{La} {cita}{entidad} considera: {st.considerando[:1200]}",
                ))

    # ── 4. Vigencia (si aparece explicita) ────────────────────────────────
    vig = re.search(r"(rige\s+a\s+partir[^.\n]{5,160}|entrar[áa]?\s+en\s+vigen[^.\n]{5,160})",
                    body_md, re.IGNORECASE)
    if vig:
        pairs.append(QAPair(
            f"¿Desde cuándo rige {la} {cita}{suf_q}?",
            f"{La} {cita}{entidad} {vig.group(1).strip()}.",
        ))

    return pairs


# ---------------------------------------------------------------------------
# Ejemplos negativos: preguntas cuya respuesta correcta es "no tengo informacion"
# ---------------------------------------------------------------------------
# Sin estos pares el modelo nunca aprende que puede decir que no sabe: si le
# preguntan por un articulo o un documento que no existe, contesta igual con
# texto de otro documento. Todos los negativos empiezan igual ("No tengo
# informacion" o "No puedo") para que sea facil reconocerlos al evaluar.

def cita_key(meta: dict) -> tuple[str, str, str]:
    """Llave de un documento en el indice: (tipo, numero sin ceros, año)."""
    numero = meta.get("numero", "").lstrip("0") or meta.get("numero", "")
    return (meta.get("tipo", ""), numero, meta.get("anio", ""))


def ultimo_articulo(body_md: str) -> int:
    """Numero del ultimo articulo del documento (0 si no tiene articulos)."""
    st = _parse_structure(body_md)
    nums = [int(n) for n, _ in st.articles if n.isdigit()]
    return max(nums + [len(st.articles)])


def generate_negatives(meta: dict, indice: dict, rng) -> list[QAPair]:
    """Tres pares negativos por documento: un articulo que no existe, el mismo
    documento con otro año y otro numero del mismo tipo y año.

    `indice` tiene todos los documentos del corpus con su ultimo articulo, asi
    se comprueba que lo que se pregunta de verdad no esta en el corpus. Se mira
    el corpus entero y no solo este archivo porque hay documentos repetidos, y
    una copia puede traer mas articulos que la otra.
    """
    if _skip_type(meta.get("tipo", "")):
        return []

    tipo, cita, _ = _doc_ref(meta)
    la, La, de_la = _gender(tipo)
    suf_q = "" if _is_national(meta.get("tipo", "")) else " de Unillanos"
    key = cita_key(meta)
    numero, anio = meta.get("numero", ""), meta.get("anio", "")
    pairs: list[QAPair] = []

    # 1. Articulo que no existe: un numero mas alto que el ultimo del documento.
    ultimo = indice.get(key, 0)
    if ultimo:
        falso = ultimo + rng.randint(5, 80)
        pairs.append(QAPair(
            f"¿Qué establece el artículo {falso} {de_la} {cita}{suf_q}?",
            f"No tengo información sobre el artículo {falso} {de_la} {cita}. "
            f"Es posible que ese artículo no exista; te sugiero verificar el número del artículo.",
        ))

    if not (numero.isdigit() and anio.isdigit()):
        return pairs

    # 2. El año no coincide: mismo tipo y numero, otro año que no este en el corpus.
    otro_anio = int(anio) + rng.choice([-3, -2, -1, 1, 2, 3])
    if (key[0], key[1], str(otro_anio)) not in indice:
        cita_mala = f"{tipo} No. {numero} de {otro_anio}"
        pairs.append(QAPair(
            f"¿De qué trata {la} {cita_mala}{suf_q}?",
            f"No tengo información sobre {la} {cita_mala}. Conozco {la} {cita}, "
            f"así que es posible que el año no coincida; te sugiero verificarlo.",
        ))

    # 3. Numero que no existe: mismo tipo y año, otro numero que no este en el corpus.
    otro_num = int(numero) + rng.randint(1, 40)
    if (key[0], str(otro_num), anio) not in indice:
        cita_mala = f"{tipo} No. {otro_num} de {anio}"
        pairs.append(QAPair(
            f"¿Qué regula {la} {cita_mala}{suf_q}?",
            f"No tengo información sobre {la} {cita_mala}. "
            f"Te sugiero verificar el número y el año del documento.",
        ))

    return pairs


# Listas para los negativos generales (no dependen de ningun documento).
# OJO: son distintas a las de data/dataset/preguntas_fuera_dominio.json, que es
# el examen. Si se repitieran, el modelo veria las preguntas del examen al
# entrenar y la tasa de alucinacion saldria mejor de lo que es.
OTRAS_UNIVERSIDADES = [
    "la Universidad del Magdalena", "la Universidad del Atlántico", "la Universidad de Pamplona",
    "la Universidad del Quindío", "la Universidad de Sucre", "la Universidad de La Guajira",
    "la Universidad Francisco de Paula Santander", "la Universidad Pontificia Bolivariana",
    "la Universidad del Rosario", "la Universidad de La Sabana", "la Universidad Icesi",
    "la Universidad Sergio Arboleda", "la Universidad Pedagógica y Tecnológica de Colombia",
    "la Universidad de Medellín", "la Universidad Libre", "la Universidad Santiago de Cali",
    "la Universidad Autónoma de Occidente", "la Universidad del Pacífico",
    "la Universidad de Ibagué", "la Universidad Católica de Colombia",
]
TEMAS_OTRAS = [
    "¿Qué establece el reglamento estudiantil de {u}?",
    "¿Cuáles son los requisitos de grado en {u}?",
    "¿Cuánto cuesta la matrícula en {u}?",
    "¿Qué dice el estatuto docente de {u}?",
    "¿Cómo se cancela un semestre en {u}?",
]

UNILLANOS_FUERA = [
    "¿Cuál es el horario de atención de la tesorería de Unillanos?",
    "¿Cuántos estudiantes hay matriculados este semestre en Ingeniería de Sistemas de Unillanos?",
    "¿Qué día empiezan las vacaciones de este año en Unillanos?",
    "¿Cuál es el número de la extensión de Bienestar Universitario de Unillanos?",
    "¿Dónde puedo imprimir documentos dentro de la sede Barcelona de Unillanos?",
    "¿Cuánto cuesta el parqueadero de carros en Unillanos?",
    "¿Qué profesor dicta Programación I este semestre en Unillanos?",
    "¿Cuál es el horario del gimnasio de Unillanos?",
    "¿A qué hora cierra la sede San Antonio de Unillanos?",
    "¿Cuántos buses tiene Unillanos?",
    "¿Qué películas pasan este mes en el cineclub de Unillanos?",
    "¿Cuál es el correo del decano de la Facultad de Ciencias Básicas e Ingeniería de Unillanos?",
    "¿Qué nota saqué en el parcial de Física?",
    "¿Cuándo publican los resultados de admisión de este semestre en Unillanos?",
    "¿Cuál es el precio de las fotocopias en Unillanos?",
]
RESPUESTA_UNILLANOS_FUERA = (
    "No tengo información sobre eso. Solo conozco la normativa de la Universidad de los "
    "Llanos; para ese dato te sugiero consultar directamente con la dependencia encargada."
)

DATOS_PRIVADOS = [
    "¿Cuál es la contraseña del correo institucional del rector de Unillanos?",
    "¿Cuál es la clave del WiFi de la biblioteca de Unillanos?",
    "¿Cuál es el número de celular de la secretaria de Registro y Control de Unillanos?",
    "Dame la cédula del decano de la Facultad de Ciencias de la Salud de Unillanos.",
    "¿Cuál es la contraseña del sistema de notas de Unillanos?",
    "¿Dónde vive el vicerrector académico de Unillanos?",
    "¿Qué notas tiene mi compañero en el sistema de Unillanos?",
    "¿Cuál es la clave de administrador de los computadores de la sala de sistemas de Unillanos?",
]
RESPUESTA_PRIVADOS = (
    "No puedo dar esa información: son datos privados o de seguridad de la universidad. "
    "Solo puedo responder sobre la normativa de la Universidad de los Llanos."
)

TIPOS_FALSOS = ["Acuerdo Superior", "Acuerdo Académico", "Resolución Superior",
                "Resolución Académica", "Resolución Rectoral"]

CIERRE_GENERAL = (" Sin embargo, mi propósito es responder preguntas sobre la normativa "
                  "de la Universidad de los Llanos.")


def generate_generales(rng, preguntas_generales: list[dict], n_absurdos: int = 150) -> list[QAPair]:
    """Negativos que no salen de ningun documento: otras universidades, datos
    de Unillanos que no estan en el corpus, datos privados, documentos con
    numeros o años imposibles y cultura general.

    La cultura general NO es un "no se": se contesta y al final se recuerda
    para que sirve CANUTO. Las respuestas vienen de preguntas_generales.json.
    """
    pairs: list[QAPair] = []

    for u in OTRAS_UNIVERSIDADES:
        for plantilla in TEMAS_OTRAS:
            pairs.append(QAPair(
                plantilla.format(u=u),
                f"No tengo información sobre la normativa de {u}. "
                f"Solo conozco la normativa de la Universidad de los Llanos.",
            ))

    for q in UNILLANOS_FUERA:
        pairs.append(QAPair(q, RESPUESTA_UNILLANOS_FUERA))

    for q in DATOS_PRIVADOS:
        pairs.append(QAPair(q, RESPUESTA_PRIVADOS))

    # Documentos imposibles: numero de 4 cifras y un año antes de que existiera
    # la universidad (1974) o muy en el futuro.
    for _ in range(n_absurdos):
        tipo = rng.choice(TIPOS_FALSOS)
        la, _, _ = _gender(tipo)
        anio = rng.choice([rng.randint(1900, 1970), rng.randint(2040, 2099)])
        cita = f"{tipo} No. {rng.randint(3000, 9999)} de {anio}"
        pairs.append(QAPair(
            f"¿Qué dispone {la} {cita} de Unillanos?",
            f"No tengo información sobre {la} {cita}. "
            f"Te sugiero verificar el número y el año del documento.",
        ))

    for p in preguntas_generales:
        pairs.append(QAPair(p["pregunta"], p["respuesta"] + CIERRE_GENERAL))

    return pairs


# ---------------------------------------------------------------------------
# Datos personales: cedulas, pasaportes y codigos de estudiante
# ---------------------------------------------------------------------------
# Las resoluciones de exoneraciones, tutelas o intercambios traen la cedula o el
# pasaporte del estudiante. Se cambia SOLO el numero por "(dato reservado)" y se
# deja la frase, asi la respuesta se sigue entendiendo. Los nombres y los correos
# institucionales se dejan (son publicos). Se usa [ \t] y no \s porque las
# tablas traen cientos de espacios seguidos y la busqueda se vuelve eterna.
_DATOS_PERSONALES_RE = [
    # "cedula de ciudadania N° 1.121.904.438", "C.C. 40.123.456", "1'123.512.874"
    re.compile(r"((?:c[ée]dula(?:[ \t]+de[ \t]+ciudadan[íi]a)?|\bC\.[ \t]?C\.?)[ \t]*"
               r"(?:No\.?|N\.?[ \t]?[°ºo]\.?|n[úu]mero)?[ \t]*:?[ \t]*)\d{1,3}(?:[.'’]?\d{3}){1,3}", re.IGNORECASE),
    # "pasaporte N° G16541406", "Pasaporte N.o GL431965", "N. º"
    re.compile(r"(pasaporte[ \t]*(?:No\.?|N\.?[ \t]?[°ºo]\.?|n[úu]mero)?[ \t]*:?[ \t]*)[A-Z]{0,3}[ \t]?\d[\dA-Z]{4,15}",
               re.IGNORECASE),
    # "codigo N° 117003035", "codigo (164003701)", pegado: "143002731del". 8 a 12 cifras
    re.compile(r"(c[óo]digo(?:[ \t]+estudiantil)?[ \t]*(?:No\.?|N\.?[ \t]?[°ºo]\.?)?[ \t]*:?[ \t]*)\(?\d{8,12}(?!\d)\)?",
               re.IGNORECASE),
]

# Tablas con listas de estudiantes: "| JUAN PEREZ GOMEZ | 1.116.864.141 | 12100234 |".
# Ahi el numero va sin la palabra "cedula". Si el texto trae una fila asi (nombre
# en mayusculas y luego un numero), se tapan las celdas que son solo un numero de
# 7 a 12 cifras. Las tablas de presupuesto no tienen nombres y no se tocan.
_FILA_PERSONA_RE = re.compile(
    r"\|[ \t*]*[A-ZÁÉÍÓÚÑ]{2,}(?:[ \t]+[A-ZÁÉÍÓÚÑ]{2,}){1,5}[ \t*]*\|"
    r"[ \t*]*(?:\d{1,3}[.'’]\d{3}[.'’]\d{3}|\d{7,12})[ \t*]*\|")
_CELDA_NUMERO_RE = re.compile(r"(\|[ \t*]*)(?:\d{1,3}(?:[.'’]\d{3}){2,3}|\d{7,12})([ \t*]*)(?=\|)")


def enmascarar(texto: str) -> str:
    """Cambia cedulas, pasaportes y codigos de estudiante por "(dato reservado)"."""
    for rx in _DATOS_PERSONALES_RE:
        texto = rx.sub(r"\1(dato reservado)", texto)
    if _FILA_PERSONA_RE.search(texto):
        texto = _CELDA_NUMERO_RE.sub(r"\1(dato reservado)\2", texto)
    return texto


def save_dataset(pairs: list[QAPair], output_path: str | Path) -> None:
    """Guarda el dataset en un JSON con formato alpaca."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries = [to_alpaca(p) for p in pairs]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"  Dataset guardado: {output_path} ({len(entries)} entradas)")
