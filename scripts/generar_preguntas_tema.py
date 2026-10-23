"""Familia B: preguntas abiertas (sin cita) para los documentos clave.

Qwen2.5-7B sin entrenar escribe 3 preguntas por articulo, como las haria un
estudiante. Solo escribe la pregunta: la respuesta es el texto real del
articulo. Las preguntas pasan por filtros y por una revision pregunta por
pregunta (campo "revision"), y se guardan en data/dataset/preguntas_tema.json
(versionado; build_dataset.py usa las "ok").

Uso (en la workstation, nunca a la vez que un entrenamiento):
    python scripts/generar_preguntas_tema.py               # genera, ~1 h de GPU
    python scripts/generar_preguntas_tema.py --refiltrar   # vuelve a filtrar sin Qwen
"""
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

from build_dataset import cargar_documentos, cargar_correcciones, cargar_modificados
from src.dataset.qa_builder import generate_heuristic, cita_key

MODELO = "Qwen/Qwen2.5-7B-Instruct"
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SALIDA = RAIZ / "data/dataset/preguntas_tema.json"
SEMILLA = 42
TOP_IDA_Y_VUELTA = 3

# Sin nivel, una pregunta se entiende de pregrado: las de posgrado lo deben decir
# y las de pregrado no pueden decirlo, o una pregunta tendria dos respuestas.
POSGRADO = {("ACUERDO_SUPERIOR", "3", "2023"), ("ACUERDO_ACADEMICO", "11", "2025")}
PREGRADO = {("ACUERDO_SUPERIOR", "20", "2021"), ("ACUERDO_ACADEMICO", "3", "2023"),
            ("ACUERDO_ACADEMICO", "16", "2025"), ("ACUERDO_SUPERIOR", "1", "2015")}
FCBI = {("RESOLUCION_CONSEJO_DE_FACULTAD_-_FCBI", "61", "2025")}
NIVEL_RE = re.compile(r"posgrado|maestr[ií]a|especializaci[oó]n|doctorado", re.IGNORECASE)
FCBI_RE = re.compile(r"FCBI|Ciencias B[aá]sicas", re.IGNORECASE)

CITA_RE = re.compile(r"art[ií]culo\s*\d|\b(acuerdo|resoluci[oó]n)\b[^?]{0,30}\d|\bde\s+(19|20)\d\d\b",
                     re.IGNORECASE)
# Preguntas que solo se entienden con el fragmento delante, o que piden opinion.
CONTEXTO_RE = re.compile(r"\b(este|esta|estos|estas|dicho|dicha)\s+(acuerdo|reglamento|estatuto|art[ií]culo|"
                         r"fragmento|texto|documento|norma|resoluci[oó]n|modificaci[oó]n|actividades|procesos|"
                         r"casos|requisitos|plan|sistema|recursos)\b|"
                         r"seg[uú]n (el|este) (acuerdo|fragmento|texto|art[ií]culo)", re.IGNORECASE)
OPINION_RE = re.compile(r"\b(tu|su) (experiencia|historia)\b|\bhas visto\b|\bha participado\b|\bcrees que\b|"
                        r"\bopinas\b|\busted ha\b|\brecomendar[ií]as\b|\bcompart[ií] un caso\b", re.IGNORECASE)
# Qwen a veces cambia de idioma (salieron preguntas en chino).
OTRO_IDIOMA_RE = re.compile(r"[^\x00-ɏ¿¡“”‘’–—…°ºª]")

PARTE_RE = re.compile(r"^\(parte \d+ de \d+\)\s*")
VIGENCIA_RE = re.compile(r"^(vigencia|el presente|la presente|rige)", re.IGNORECASE)

INSTRUCCION = """Este es un fragmento {de_nombre} de la Universidad de los Llanos:

"{texto}"

Escribe 3 preguntas distintas que haría un estudiante o alguien de la universidad, cuya respuesta esté en este fragmento.
- No menciones el número del artículo ni del acuerdo o la resolución.
- Escríbelas como habla la gente: unas de tú, otras de usted; alguna puede contar su caso.
- Una pregunta por línea, sin numerar y sin nada más.{extra}"""


def cargar_clave(ruta: Path) -> dict:
    """Lee config/documentos_clave.txt: (tipo, numero, año) -> nombre comun."""
    clave = {}
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.split("#")[0]
        if "|" in linea:
            cita, nombre = linea.split("|")
            clave[tuple(cita.split())] = nombre.strip()
    return clave


def _de(nombre: str) -> str:
    """'de' + nombre, con la contraccion: 'el Estatuto' -> 'del Estatuto'."""
    return "del " + nombre[3:] if nombre.startswith("el ") else "de " + nombre


def piezas():
    """Articulos, paragrafos y partes de los documentos clave y de los que los
    modificaron. "cita" (la pregunta A1) es la llave que usa build_dataset.py."""
    correcciones = cargar_correcciones(RAIZ / "config/correcciones_citas.tsv")
    clave = cargar_clave(RAIZ / "config/documentos_clave.txt")
    modificados = cargar_modificados(RAIZ / "config/articulos_modificados.tsv")
    modificadores = {}   # documento que modifico -> documento clave que modifico
    for doc, arts in modificados.items():
        for por in arts.values():
            modificadores[por] = doc

    resultado = []
    docs, _, _ = cargar_documentos(sorted((RAIZ / "data/extracted").rglob("*.md")), correcciones)
    for _, meta, body in docs:
        key = cita_key(meta)
        viejos = modificados.get(key, {})   # articulos de este documento que ya cambiaron
        if key in clave:
            nombre = clave[key]
        elif key in modificadores:   # hereda el nivel del documento clave
            nombre = f"un acuerdo que modifica {clave[modificadores[key]]}"
            key = modificadores[key]
        else:
            continue

        vistas = set()   # la primera pregunta de cada respuesta es la A1
        for par in generate_heuristic(body, meta):
            if par.answer in vistas or "se establece:" not in par.answer:
                continue
            vistas.add(par.answer)
            texto = par.answer.split("se establece:", 1)[1].strip()
            # Los articulos modificados salen del documento que los cambio.
            articulo = re.search(r"artículo (\S+) d", par.question).group(1)
            if VIGENCIA_RE.search(PARTE_RE.sub("", texto)) or articulo in viejos:
                continue
            resultado.append({"cita": par.question, "respuesta": par.answer, "texto": texto,
                              "nombre": nombre, "doc": key})
    return resultado


def cargar_qwen():
    """En 4 bits, como en la inferencia, para que quepa en los 16 GB."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    torch.manual_seed(SEMILLA)
    tokenizer = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO, quantization_config=quant, device_map="cuda")
    return modelo, tokenizer


def instruccion(pieza):
    extra = ""
    if pieza["doc"] in POSGRADO:
        extra = "\n- Cada pregunta debe decir que es de posgrado, maestría, especialización o doctorado."
    elif pieza["doc"] in FCBI:
        extra = "\n- Cada pregunta debe mencionar la Facultad de Ciencias Básicas e Ingeniería."
    return INSTRUCCION.format(de_nombre=_de(pieza["nombre"]), texto=PARTE_RE.sub("", pieza["texto"]), extra=extra)


def preguntar(modelo, tokenizer, pieza):
    import torch
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": instruccion(pieza)}],
                                           tokenize=False, add_generation_prompt=True)
    entrada = tokenizer([prompt], return_tensors="pt").to("cuda")
    with torch.no_grad():   # con temperatura, para que las 3 salgan distintas
        salida = modelo.generate(**entrada, max_new_tokens=200, do_sample=True,
                                 temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    generado = tokenizer.decode(salida[0][entrada["input_ids"].shape[-1]:], skip_special_tokens=True)
    # Qwen numera a veces aunque se le pida que no.
    return [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", l).strip() for l in generado.splitlines() if l.strip()]


def filtro(pregunta, doc):
    """"ok" o la razon del descarte. Vale empezar contando el caso: "Si..., ¿...?"."""
    if not ("¿" in pregunta and pregunta.endswith("?") and 15 <= len(pregunta) <= 250):
        return "forma"
    if OTRO_IDIOMA_RE.search(pregunta):
        return "otro idioma"
    if CONTEXTO_RE.search(pregunta):
        return "depende del fragmento"
    if OPINION_RE.search(pregunta):
        return "pide opinion"
    if CITA_RE.search(pregunta):
        return "trae cita"
    if doc in POSGRADO and not NIVEL_RE.search(pregunta):
        return "posgrado sin nivel"
    if doc in PREGRADO and NIVEL_RE.search(pregunta):
        return "pregrado con nivel de posgrado"
    if doc in FCBI and not FCBI_RE.search(pregunta):
        return "FCBI sin facultad"
    return "ok"


def quitar_choques(preguntas):
    """La misma pregunta para dos piezas distintas es una contradiccion."""
    piezas_de = {}
    for p in preguntas:
        piezas_de.setdefault(p["pregunta"].lower(), set()).add(p["cita"])
    for p in preguntas:
        if p["filtro"] == "ok" and len(piezas_de[p["pregunta"].lower()]) > 1:
            p["filtro"] = "choque"


def ida_y_vuelta(preguntas, lista_piezas):
    """Su pieza debe estar entre las TOP_IDA_Y_VUELTA mas parecidas del mismo
    documento. Entre documentos ya decide la regla de nivel."""
    from sentence_transformers import SentenceTransformer, util
    emb = SentenceTransformer(MODELO_EMBEDDINGS, device="cpu")
    doc_de = {p["cita"]: p["doc"] for p in lista_piezas}
    citas = [p["cita"] for p in lista_piezas]
    v_piezas = emb.encode([p["texto"] for p in lista_piezas], convert_to_tensor=True)
    ok = [p for p in preguntas if p["filtro"] == "ok"]
    v_preg = emb.encode([p["pregunta"] for p in ok], convert_to_tensor=True)
    for p, sim in zip(ok, util.cos_sim(v_preg, v_piezas)):
        mismo_doc = [i for i in sim.argsort(descending=True).tolist() if doc_de[citas[i]] == doc_de[p["cita"]]]
        if p["cita"] not in [citas[i] for i in mismo_doc[:TOP_IDA_Y_VUELTA]]:
            p["filtro"] = "ambigua"


def guardar_y_contar(preguntas):
    with open(SALIDA, "w", encoding="utf-8") as f:
        json.dump(preguntas, f, ensure_ascii=False, indent=2)
    razones = {}
    for p in preguntas:
        razones[p["filtro"]] = razones.get(p["filtro"], 0) + 1
    print(f"Preguntas: {len(preguntas)} -> {SALIDA}")
    for razon, n in sorted(razones.items(), key=lambda x: -x[1]):
        print(f"  {n:5d}  {razon}")


def refiltrar():
    lista = piezas()
    doc_de = {p["cita"]: p["doc"] for p in lista}
    with open(SALIDA, encoding="utf-8") as f:
        preguntas = json.load(f)
    for p in preguntas:
        p["filtro"] = filtro(p["pregunta"], doc_de[p["cita"]]) if p["cita"] in doc_de else "sin pieza"
    quitar_choques(preguntas)
    ida_y_vuelta(preguntas, lista)
    aplicar_revision(preguntas)
    guardar_y_contar(preguntas)


def aplicar_revision(preguntas):
    """Las preguntas rechazadas en la revision no vuelven a ser "ok" al refiltrar."""
    for p in preguntas:
        if p["filtro"] == "ok" and p.get("revision") == "no":
            p["filtro"] = "revision dijo no"


def main():
    if "--refiltrar" in sys.argv:
        refiltrar()
        return
    if SALIDA.exists():
        print(f"Ya existe {SALIDA}: esta revisado a mano. Borralo a proposito si quieres regenerarlo.")
        sys.exit(1)

    lista = piezas()
    print(f"Piezas de documentos clave: {len(lista)}")

    modelo, tokenizer = cargar_qwen()

    preguntas = []
    for i, pieza in enumerate(lista, 1):
        for pregunta in preguntar(modelo, tokenizer, pieza)[:3]:
            preguntas.append({"pregunta": pregunta, "cita": pieza["cita"],
                              "filtro": filtro(pregunta, pieza["doc"])})
        print(f"  {i}/{len(lista)}", end="\r")
    print()
    del modelo   # se suelta la GPU antes de los embeddings

    quitar_choques(preguntas)
    ida_y_vuelta(preguntas, lista)
    guardar_y_contar(preguntas)


if __name__ == "__main__":
    main()
