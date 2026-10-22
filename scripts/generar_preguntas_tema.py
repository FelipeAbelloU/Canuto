"""Familia B: preguntas abiertas (sin cita) para los documentos clave.

Qwen2.5-7B SIN entrenar lee cada articulo de los documentos clave y escribe 3
preguntas como las haria un estudiante ("¿Que nota necesito para homologar?").
Qwen solo escribe PREGUNTAS: la respuesta es siempre el texto real del articulo,
la misma de las preguntas con cita, asi que no puede inventar nada.

Las preguntas pasan por filtros (forma, sin cita, nivel pregrado/posgrado, sin
choques entre articulos y que apunten a su articulo) y se guardan en
data/dataset/preguntas_tema.json, que se versiona en git y se revisa a mano.
build_dataset.py usa solo las que tienen "filtro": "ok".

Corre en la workstation (GPU). Nunca a la vez que un entrenamiento: la tarjeta
es compartida. Tarda ~1 hora.

Uso:
    python scripts/generar_preguntas_tema.py
    python scripts/generar_preguntas_tema.py --refiltrar   # vuelve a filtrar sin Qwen
"""
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

from build_dataset import _split_frontmatter, cargar_correcciones, cargar_clave, cargar_modificados
from src.dataset.qa_builder import generate_heuristic, cita_key

MODELO = "Qwen/Qwen2.5-7B-Instruct"
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SALIDA = RAIZ / "data/dataset/preguntas_tema.json"
SEMILLA = 42
TOP_IDA_Y_VUELTA = 3   # el articulo de la pregunta debe quedar entre los 3 mas parecidos

# Regla de nivel. Sin nivel, la pregunta se entiende de pregrado; por eso las de
# posgrado lo deben decir y las de pregrado no pueden decir posgrado. Si no, la
# misma pregunta tendria dos respuestas (los dos reglamentos dicen cosas distintas).
POSGRADO = {("ACUERDO_SUPERIOR", "3", "2023"), ("ACUERDO_ACADEMICO", "11", "2025")}
PREGRADO = {("ACUERDO_SUPERIOR", "20", "2021"), ("ACUERDO_ACADEMICO", "3", "2023"),
            ("ACUERDO_ACADEMICO", "16", "2025"), ("ACUERDO_SUPERIOR", "1", "2015")}
FCBI = {("RESOLUCION_CONSEJO_DE_FACULTAD_-_FCBI", "61", "2025")}
NIVEL_RE = re.compile(r"posgrado|maestr[ií]a|especializaci[oó]n|doctorado", re.IGNORECASE)
FCBI_RE = re.compile(r"FCBI|Ciencias B[aá]sicas", re.IGNORECASE)

# Una pregunta abierta no trae cita: ni "articulo 46" ni "Acuerdo 20 de 2021".
CITA_RE = re.compile(r"art[ií]culo\s*\d|\b(acuerdo|resoluci[oó]n)\b[^?]{0,30}\d|\bde\s+(19|20)\d\d\b",
                     re.IGNORECASE)
# Preguntas que solo se entienden leyendo el fragmento ("segun este acuerdo",
# "estas actividades") o que le piden opinion o experiencia al bot.
CONTEXTO_RE = re.compile(r"\b(este|esta|estos|estas|dicho|dicha)\s+(acuerdo|reglamento|estatuto|art[ií]culo|"
                         r"fragmento|texto|documento|norma|resoluci[oó]n|modificaci[oó]n|actividades|procesos|"
                         r"casos|requisitos|plan|sistema|recursos)\b|"
                         r"seg[uú]n (el|este) (acuerdo|fragmento|texto|art[ií]culo)", re.IGNORECASE)
OPINION_RE = re.compile(r"\b(tu|su) (experiencia|historia)\b|\bhas visto\b|\bha participado\b|\bcrees que\b|"
                        r"\bopinas\b|\busted ha\b|\brecomendar[ií]as\b|\bcompart[ií] un caso\b", re.IGNORECASE)
# Qwen a veces se pasa a otro idioma (salieron preguntas en chino).
OTRO_IDIOMA_RE = re.compile(r"[^\x00-ɏ¿¡“”‘’–—…°ºª]")

# Articulos de vigencia o derogatorias: nadie pregunta por ellos.
VIGENCIA_RE = re.compile(r"^(\(parte \d+ de \d+\)\s*)?(vigencia|el presente|la presente|rige)", re.IGNORECASE)

INSTRUCCION = """Este es un fragmento {de_nombre} de la Universidad de los Llanos:

"{texto}"

Escribe 3 preguntas distintas que haría un estudiante o alguien de la universidad, cuya respuesta esté en este fragmento.
- No menciones el número del artículo ni del acuerdo o la resolución.
- Escríbelas como habla la gente: unas de tú, otras de usted; alguna puede contar su caso.
- Una pregunta por línea, sin numerar y sin nada más.{extra}"""


def piezas():
    """Las piezas (articulo, paragrafo o parte) de los documentos clave, mas los
    articulos de los documentos que modificaron a alguno de ellos.

    Cada pieza: {"cita": pregunta A1, "respuesta", "texto", "nombre", "doc"}. La
    pregunta A1 es la llave con la que build_dataset.py encuentra la respuesta.
    """
    correcciones = cargar_correcciones(RAIZ / "config/correcciones_citas.tsv")
    clave = cargar_clave(RAIZ / "config/documentos_clave.txt")
    modificados = cargar_modificados(RAIZ / "config/articulos_modificados.tsv")
    modificadores = {}   # documento que modifico -> documento clave que modifico
    for doc, arts in modificados.items():
        for por in arts.values():
            modificadores[por] = doc

    resultado = []
    for ruta in sorted((RAIZ / "data/extracted").rglob("*.md")):
        meta, body = _split_frontmatter(ruta.read_text(encoding="utf-8"))
        fila = correcciones.get(meta.get("ruta_origen", ""))
        if fila:
            if fila["accion"] == "excluir":
                continue
            meta["numero"], meta["anio"] = fila["numero"], fila["anio"]
        key = cita_key(meta)
        viejos = modificados.get(key, {})   # articulos de este documento que ya cambiaron
        if key in clave:
            nombre = clave[key]
        elif key in modificadores:
            # Se toma el nivel del documento clave que modifica.
            nombre = f"un acuerdo que modifica {clave[modificadores[key]]}"
            key = modificadores[key]
        else:
            continue

        # La primera pregunta de cada respuesta es la A1 (con la cita formal).
        vistas = set()
        for par in generate_heuristic(body, meta):
            if par.answer in vistas or "se establece:" not in par.answer:
                continue
            vistas.add(par.answer)
            texto = par.answer.split("se establece:", 1)[1].strip()
            # Los articulos viejos (modificados) no: salen del documento que los cambio.
            articulo = re.search(r"artículo (\S+) d", par.question).group(1)
            if VIGENCIA_RE.search(texto) or articulo in viejos:
                continue
            resultado.append({"cita": par.question, "respuesta": par.answer, "texto": texto,
                              "nombre": nombre, "doc": key})
    return resultado


def cargar_qwen():
    """Qwen2.5-7B en 4 bits, como en la inferencia, para que quepa en los 16 GB."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO, quantization_config=quant, device_map="cuda")
    return modelo, tokenizer


def instruccion(pieza):
    """Arma lo que se le pide a Qwen para una pieza."""
    extra = ""
    if pieza["doc"] in POSGRADO:
        extra = "\n- Cada pregunta debe decir que es de posgrado, maestría, especialización o doctorado."
    elif pieza["doc"] in FCBI:
        extra = "\n- Cada pregunta debe mencionar la Facultad de Ciencias Básicas e Ingeniería."
    nombre = pieza["nombre"]
    de_nombre = "del " + nombre[3:] if nombre.startswith("el ") else "de " + nombre
    fragmento = re.sub(r"^\(parte \d+ de \d+\)\s*", "", pieza["texto"])
    return INSTRUCCION.format(de_nombre=de_nombre, texto=fragmento, extra=extra)


def preguntar(modelo, tokenizer, pieza):
    """Le pide a Qwen 3 preguntas sobre la pieza y devuelve las lineas que escribio."""
    import torch
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": instruccion(pieza)}],
                                           tokenize=False, add_generation_prompt=True)
    entrada = tokenizer([prompt], return_tensors="pt").to("cuda")
    # Con temperatura las 3 preguntas salen distintas entre si.
    with torch.no_grad():
        salida = modelo.generate(**entrada, max_new_tokens=200, do_sample=True,
                                 temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    generado = tokenizer.decode(salida[0][entrada["input_ids"].shape[-1]:], skip_special_tokens=True)
    # Se quitan numeraciones o viñetas que Qwen ponga a pesar de la instruccion.
    return [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", l).strip() for l in generado.splitlines() if l.strip()]


def filtro(pregunta, doc):
    """Devuelve "ok" o la razon por la que se descarta (filtros 1 a 3).

    La pregunta puede empezar contando el caso ("Si perdi una materia, ¿puedo...?"):
    basta con que tenga el "¿" y termine en "?".
    """
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
    """Filtro 4: la misma pregunta para dos piezas distintas es una contradiccion."""
    piezas_de = {}
    for p in preguntas:
        piezas_de.setdefault(p["pregunta"].lower(), set()).add(p["cita"])
    for p in preguntas:
        if p["filtro"] == "ok" and len(piezas_de[p["pregunta"].lower()]) > 1:
            p["filtro"] = "choque"


def ida_y_vuelta(preguntas, lista_piezas):
    """Filtro 5: la pregunta debe parecerse mas a su pieza que a las otras del
    MISMO documento. Si su pieza no queda entre las TOP_IDA_Y_VUELTA mas
    parecidas, es ambigua. Se compara dentro del documento porque entre
    documentos (pregrado y posgrado tienen articulos casi iguales) ya decide la
    regla de nivel; comparando contra todos se botaban preguntas claras."""
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
    """Vuelve a pasar los filtros a las preguntas ya generadas, sin Qwen ni GPU."""
    lista = piezas()
    doc_de = {p["cita"]: p["doc"] for p in lista}
    with open(SALIDA, encoding="utf-8") as f:
        preguntas = json.load(f)
    for p in preguntas:
        p["filtro"] = filtro(p["pregunta"], doc_de[p["cita"]]) if p["cita"] in doc_de else "sin pieza"
    quitar_choques(preguntas)
    ida_y_vuelta(preguntas, lista)
    guardar_y_contar(preguntas)


def main():
    if "--refiltrar" in sys.argv:
        refiltrar()
        return
    if SALIDA.exists():
        print(f"Ya existe {SALIDA}: esta revisado a mano. Borralo a proposito si quieres regenerarlo.")
        sys.exit(1)

    lista = piezas()
    print(f"Piezas de documentos clave: {len(lista)}")

    import torch
    torch.manual_seed(SEMILLA)
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
