"""Evalua un checkpoint sobre el conjunto de prueba (data/dataset/test.json).

El script hace dos cosas SEPARADAS:

  1. GENERAR (lo caro, ~45 min): le pasa cada pregunta al modelo y guarda las
     respuestas en data/eval/<nombre-del-checkpoint>.json. De paso calcula, con
     el modelo ya cargado, la perdida y la entropia de cada ejemplo.
  2. MEDIR (segundos): lee ese archivo y calcula las metricas.

Si el archivo de respuestas ya existe se reutiliza, asi que corregir o agregar
una metrica cuesta segundos en vez de volver a generar. Con --regenerar se
fuerza a generar de nuevo.

Metricas:
  - Exactitud de citas y tasa de alucinacion: son las que deciden el ganador.
  - BERTScore-F1, coseno semantico, F1 por tokens: parecido con la referencia.
  - ROUGE-L, BLEU, CIDEr: solapamiento de texto (comparan corridas entre si).
  - Perdida, perplejidad, entropia, informacion mutua: intrinsecas.

Al final imprime una linea con las columnas separadas por tabulacion, lista
para pegar en el Excel de la bitacora.

Uso:
    python scripts/evaluate.py --checkpoint data/checkpoints/t1_7b-e3-lr2e4-r16-wd0.0
    python scripts/evaluate.py --checkpoint <carpeta> --regenerar
    python scripts/evaluate.py --checkpoint <carpeta> --limit 10   # solo para probar el script
"""
import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config_loader import load_config
from src.factory import create_model

# Fijos aqui y no en config.yaml (que comparte la web): cambiarlos rompe la
# comparacion entre corridas.
MAX_NEW_TOKENS = 1024
TEMPERATURA = 0           # greedy: la misma pregunta da siempre la misma respuesta
SEQ_LEN = 2048            # el mismo del entrenamiento

# Muestra fija del test por familia: las primeras 150 traian 1 sola B y ninguna E.
# La E tiene 26 en el test v2: entran todas.
MUESTRA = {"A": 60, "B": 40, "E": 26, "negativo": 60}
SEMILLA_MUESTRA = 2026

MODELO_EMBEDDINGS ="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Abstencion por palabras clave, no un clasificador: "no tengo datos, pero la
# Resolucion 99 dice..." cuenta como no alucinada (decirlo en la tesis).
FRASES_NO_SE = [
    "no tengo", "no dispongo", "no cuento con", "no encuentro", "no aparece",
    "no está en", "no se encuentra", "no hay información", "no tengo información",
    "no tengo certeza", "no forma parte", "no puedo", "no está disponible",
    "no corresponde", "fuera de mi", "no existe", "no figura", "no consta",
    # halladas al revisar a mano la corrida 1
    "no se ha establecido", "no está especificado", "no se especifica",
    "no tengo acceso", "no se ha definido", "no se menciona",
    "puede variar", "podría variar", "puede cambiar",
    "página oficial", "sitio web oficial", "página web de",
    "póngase en contacto", "ponerse en contacto", "comuníquese con",
    "te recomiendo consultar", "le recomiendo consultar", "recomiendo verificar",
    "consultar directamente", "verificar directamente", "contactar directamente",
]

# Citas con "No. 10 de 2015" o sin el "No."; sin "No." se exige el tipo delante,
# o cualquier "2 de 2023" suelto contaria como cita.
CITA_RE = re.compile(r"N[°ºo\.]*\s*0*(\d{1,4})\s+de\s+(\d{4})", re.IGNORECASE)
TIPOS_DOC = r"acuerdo|resoluci[oó]n|decreto|estatuto|reglamento|circular|ley|acta"
CITA_SIN_NO_RE = re.compile(
    rf"(?:{TIPOS_DOC})\s+(?:\w+\s+)?0*(\d{{1,4}})\s+de\s+(\d{{4}})", re.IGNORECASE)

# Bloque de control de las corridas t1 y t2 (ya no esta en el archivo actual).
BLOQUE_CONTROL = "conocimiento_general"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluacion de un checkpoint de CANUTO")
    p.add_argument("--checkpoint", default=None,
                   help="Carpeta del adaptador. Si no se pasa, usa la de config.yaml")
    p.add_argument("--test", default=str(RAIZ / "data/dataset/test.json"))
    p.add_argument("--fuera-dominio",
                   default=str(RAIZ / "data/dataset/preguntas_fuera_dominio.json"))
    p.add_argument("--config", default=str(RAIZ / "config/config.yaml"))
    p.add_argument("--limit", type=int, default=None,
                   help="Solo las primeras N de la muestra, para probar el script")
    p.add_argument("--regenerar", action="store_true",
                   help="Vuelve a generar las respuestas aunque ya existan")
    return p.parse_args()


# Generar las respuestas (necesita el modelo).
def perdida_entropia(hf_model, tokenizer, pregunta, respuesta, device, system_prompt):
    """Un forward pass sobre (pregunta + respuesta de referencia), en el MISMO
    formato con el que entrena train_gpu.py (system prompt + user/assistant).

    Devuelve (perdida, entropia) sobre los tokens de la respuesta. La perplejidad
    se calcula despues como exp(media de las perdidas), que es mas estable. Se
    trunca a SEQ_LEN para no medir fuera de la distribucion de entrenamiento ni
    reventar la VRAM con respuestas largas.
    """
    import torch

    base = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": pregunta}]
    completo = base + [{"role": "assistant", "content": respuesta}]
    texto_prompt = tokenizer.apply_chat_template(base, tokenize=False, add_generation_prompt=True)
    texto_completo = tokenizer.apply_chat_template(completo, tokenize=False,
                                                   add_generation_prompt=False)

    # add_special_tokens=False: la plantilla ya trae los tokens especiales como
    # texto, asi el prompt tokeniza igual que el prefijo del texto completo.
    n_prompt = tokenizer(texto_prompt, add_special_tokens=False,
                         return_tensors="pt").input_ids.shape[1]
    destino = "cpu" if device == "cpu" else "cuda"
    ids = tokenizer(texto_completo, add_special_tokens=False,
                    return_tensors="pt").input_ids[:, :SEQ_LEN].to(destino)

    with torch.no_grad():
        logits = hf_model(ids).logits[0, :-1, :]   # predice el token t desde t-1
    labels = ids[0, 1:]
    ini = max(n_prompt - 1, 0)                     # solo cuentan los de la respuesta
    sel_logits, sel_labels = logits[ini:], labels[ini:]
    if sel_labels.numel() == 0:
        return 0.0, 0.0
    logprobs = torch.log_softmax(sel_logits.float(), dim=-1)
    nll = -logprobs[range(sel_labels.shape[0]), sel_labels].mean().item()
    entropia = -(logprobs.exp() * logprobs).sum(dim=-1).mean().item()
    return nll, entropia


def guardar(ruta, datos):
    """Escribe el archivo de respuestas. Se llama dos veces: apenas terminan las
    generaciones y otra vez con la perdida ya calculada."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)


def muestra_por_familia(test):
    rng = random.Random(SEMILLA_MUESTRA)
    elegidas = []
    for familia, cuantas in MUESTRA.items():
        de_la_familia = [r for r in test if r.get("familia") == familia]
        elegidas += rng.sample(de_la_familia, min(cuantas, len(de_la_familia)))
    return elegidas


def generar(ruta_salida, checkpoint, config, test, fuera, limite):
    """Genera las respuestas del modelo y las guarda en un JSON."""
    modelo = create_model(config, checkpoint=checkpoint)
    if modelo is None or not modelo.is_available():
        print("No hay checkpoint. Pasalo con --checkpoint o ponlo en config.yaml.")
        sys.exit(1)

    modelo.temperature = TEMPERATURA
    modelo.max_new_tokens = MAX_NEW_TOKENS
    print(f"Modelo: {modelo.checkpoint_path} (device: {modelo.device})")

    print(f"\nGenerando respuestas de {len(test)} preguntas de prueba...")
    casos = []
    for i, r in enumerate(test, 1):
        respuesta = modelo.chat([{"role": "user", "content": r["instruction"]}])
        casos.append({"pregunta": r["instruction"], "referencia": r["output"],
                      "familia": r.get("familia"), "respuesta": respuesta})
        print(f"  {i}/{len(test)}", end="\r")
    print()

    print(f"Generando respuestas de {len(fuera)} preguntas fuera de dominio...")
    casos_fuera = []
    for i, p in enumerate(fuera, 1):
        respuesta = modelo.chat([{"role": "user", "content": p["pregunta"]}])
        casos_fuera.append({"pregunta": p["pregunta"], "bloque": p["bloque"],
                            "respuesta": respuesta})
        print(f"  {i}/{len(fuera)}", end="\r")
    print()

    datos = {
        "checkpoint": Path(modelo.checkpoint_path).name,
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_test": len(casos),
        "limit": limite,
        "muestra": MUESTRA,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperatura": TEMPERATURA,
        "casos": casos,
        "fuera_dominio": casos_fuera,
    }

    # Se guarda ya: si falla la perdida, no se pierde media hora de GPU.
    guardar(ruta_salida, datos)
    print(f"Respuestas guardadas en: {ruta_salida}")

    print("Calculando perdida y entropia...")
    for i, caso in enumerate(casos, 1):
        nll, ent = perdida_entropia(modelo.hf_model, modelo.tokenizer,
                                    caso["pregunta"], caso["referencia"], modelo.device,
                                    modelo.system_prompt)
        caso["perdida"] = nll
        caso["entropia"] = ent
        print(f"  {i}/{len(casos)}", end="\r")
    print()

    guardar(ruta_salida, datos)

    # BERTScore y los embeddings no caben junto al 7B en 16 GB.
    del modelo
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    return datos


# Carpeta de train_gpu.py: v2t1_7b-e4-lr2e4-r16-wd0.01-do0.05-wu0.1. El v2, dropout y
# warmup son opcionales para leer los checkpoints viejos (t1_7b-e3-lr2e4-r16-wd0.0).
NOMBRE_RE = re.compile(
    r"^(?:v2)?t(\d+)_.*-e(\d+)-lr(\d)e(\d)-r(\d+)-wd([\d.]+)"
    r"(?:-do([\d.]+))?(?:-wu([\d.]+))?$")


def hiperparametros(carpeta):
    """Recupera los hiperparametros con los que se entreno este adaptador.

    Se busca en tres sitios, en este orden:
      1. hiperparametros.json  -> lo escribe train_gpu.py, es el registro completo
      2. adapter_config.json   -> lo escribe PEFT, manda en rank/alpha/dropout
      3. el nombre de la carpeta -> ultimo recurso para checkpoints viejos
    Lo que no se encuentre queda en None y se avisa por pantalla, para no pegar
    una fila con columnas vacias sin darse cuenta.
    """
    carpeta = Path(carpeta)
    datos = {}

    ruta = carpeta / "hiperparametros.json"
    if ruta.exists():
        with open(ruta, encoding="utf-8") as f:
            datos.update(json.load(f))

    # PEFT escribe lo que realmente se entreno: manda sobre lo demas.
    ruta_peft = carpeta / "adapter_config.json"
    if ruta_peft.exists():
        with open(ruta_peft, encoding="utf-8") as f:
            peft = json.load(f)
        datos["rank"] = peft.get("r", datos.get("rank"))
        datos["lora_alpha"] = peft.get("lora_alpha", datos.get("lora_alpha"))
        datos["lora_dropout"] = peft.get("lora_dropout", datos.get("lora_dropout"))

    # Ultimo recurso: deducirlos del nombre.
    m = NOMBRE_RE.match(carpeta.name)
    if m:
        del_nombre = {
            "num_prueba": int(m.group(1)),
            "epocas": int(m.group(2)),
            # via texto y no con 10**-exp: la multiplicacion da 0.00030000000000000003
            "learning_rate": float(f"{m.group(3)}e-{m.group(4)}"),
            "rank": int(m.group(5)),
            "weight_decay": float(m.group(6)),
        }
        if m.group(7) is not None:
            del_nombre["lora_dropout"] = float(m.group(7))
        if m.group(8) is not None:
            del_nombre["warmup_ratio"] = float(m.group(8))
        for k, v in del_nombre.items():
            datos.setdefault(k, v)

    return datos


# Metricas (solo texto, sin el modelo).
def citas(texto):
    """Conjunto de citas (numero, anio) que aparecen en un texto.

    Se quitan los ceros de la izquierda para que "007 de 1990" y "7 de 1990"
    cuenten como la misma cita.
    """
    encontradas = CITA_RE.findall(texto) + CITA_SIN_NO_RE.findall(texto)
    return {(n.lstrip("0") or "0", a) for n, a in encontradas}


def exactitud_citas(casos):
    """De las respuestas de referencia que citan un documento, en cuantas el
    modelo menciona esa misma cita."""
    con_cita = aciertos = 0
    for c in casos:
        objetivo = citas(c["referencia"])
        if objetivo:
            con_cita += 1
            if objetivo & citas(c["respuesta"]):
                aciertos += 1
    return (aciertos / con_cita) if con_cita else 0.0


def reconoce_no_saber(texto):
    t = texto.lower()
    return any(f in t for f in FRASES_NO_SE)


def tasa_alucinacion(casos_fuera):
    """Sobre preguntas de cosas que NO estan en el corpus: que fraccion contesta
    como si supiera, en vez de reconocer que no tiene la informacion.

    Las de conocimiento general NO cuentan aqui. Son preguntas de control cuya
    respuesta correcta SI existe ("Paris es la capital de Francia"): contestarlas
    bien es lo que se espera, no una alucinacion. Antes entraban al denominador
    y empujaban la tasa hacia arriba sin motivo.
    """
    evaluables = [c for c in casos_fuera if c.get("bloque") != BLOQUE_CONTROL]
    if not evaluables:
        return 0.0
    alucina = sum(0 if reconoce_no_saber(c["respuesta"]) else 1 for c in evaluables)
    return alucina / len(evaluables)


def alucinacion_por_bloque(casos_fuera):
    """La misma tasa, separada por tipo de pregunta.

    Sirve para ver DONDE falla: no es lo mismo inventarse una norma de Unillanos
    que no existe que confundirse con otra universidad.
    """
    por_bloque = {}
    for c in casos_fuera:
        bloque = c.get("bloque", "sin_bloque")
        if bloque == BLOQUE_CONTROL:
            continue
        total, alucina = por_bloque.get(bloque, (0, 0))
        por_bloque[bloque] = (total + 1,
                              alucina + (0 if reconoce_no_saber(c["respuesta"]) else 1))
    return {b: round(a / t, 4) for b, (t, a) in sorted(por_bloque.items())}


def tasa_abstencion(casos_fuera):
    """El complemento de la tasa de alucinacion: cuantas veces reconoce que no
    sabe. Es una de las dos accuracies reales del proyecto."""
    return round(1 - tasa_alucinacion(casos_fuera), 4)


def por_familia(casos):
    """Exactitud de citas y cuantas veces dice que no sabe, por familia. En los
    negativos lo bueno es que diga que no sabe; en A, B y E, que no lo diga."""
    resultado = {}
    for familia in MUESTRA:
        grupo = [c for c in casos if c.get("familia") == familia]
        if not grupo:
            continue
        no_sabe = sum(1 for c in grupo if reconoce_no_saber(c["respuesta"]))
        resultado[familia] = {
            "n": len(grupo),
            "exactitud_citas": None if familia == "negativo" else round(exactitud_citas(grupo), 4),
            "dice_no_saber": round(no_sabe / len(grupo), 4),
        }
    return resultado


def repeticion_ngramas(texto, n=4):
    """Que fraccion de los grupos de n palabras seguidas esta repetida.

    Caza el texto degenerado, que es cuando el modelo entra en bucle y repite
    la misma frase una y otra vez. Ninguna de las otras metricas lo detecta de
    frente: la entropia baja apenas lo insinua. 0 = nada repetido.
    """
    tokens = texto.lower().split()
    grupos = ngramas(tokens, n)
    if not grupos:
        return 0.0
    return 1 - len(set(grupos)) / len(grupos)


def razon_longitud(respuesta, referencia):
    """Largo de la respuesta dividido por el largo de la referencia.

    1.0 es lo ideal. Muy por debajo = se corto o no contesto; muy por encima =
    se fue por las ramas o entro en bucle. Se reporta la MEDIANA porque unos
    pocos casos extremos arrastrarian el promedio.
    """
    if not referencia.strip():
        return 0.0
    return len(respuesta.split()) / len(referencia.split())


def mediana(valores):
    ordenados = sorted(valores)
    n = len(ordenados)
    if n == 0:
        return 0.0
    if n % 2:
        return ordenados[n // 2]
    return (ordenados[n // 2 - 1] + ordenados[n // 2]) / 2


def f1_tokens(pred, ref):
    """F1 clasico de QA: solapamiento de palabras entre respuesta y referencia."""
    tp, tr = pred.lower().split(), ref.lower().split()
    if not tp or not tr:
        return 0.0
    comunes = Counter(tp) & Counter(tr)
    n = sum(comunes.values())
    if n == 0:
        return 0.0
    precision, recall = n / len(tp), n / len(tr)
    return 2 * precision * recall / (precision + recall)


def ngramas(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def cider(preds, refs, n_max=4):
    """CIDEr simplificado: coseno de vectores de n-gramas ponderados por TF-IDF,
    promediado para n=1..4. El IDF sale del corpus de referencias.

    Con una sola referencia por pregunta pierde su gracia original (medir el
    consenso entre varias referencias) y queda como un coseno TF-IDF de
    n-gramas. Se reporta porque la asesora la pidio, descrita asi de honesta.
    """
    refs_tok = [r.lower().split() for r in refs]
    preds_tok = [p.lower().split() for p in preds]
    total = len(refs_tok)
    if total == 0:
        return 0.0

    def idf(g, df):
        return math.log((total + 1) / (df.get(g, 0) + 1)) + 1

    def coseno(a, b):
        if not a or not b:
            return 0.0
        punto = sum(v * b.get(g, 0) for g, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return punto / (na * nb) if na and nb else 0.0

    puntajes = []
    for n in range(1, n_max + 1):
        refs_ng = [Counter(ngramas(rt, n)) for rt in refs_tok]
        df = Counter()
        for ng in refs_ng:
            for g in ng:
                df[g] += 1
        s = 0.0
        for pt, rng in zip(preds_tok, refs_ng):
            vp = {g: c * idf(g, df) for g, c in Counter(ngramas(pt, n)).items()}
            vr = {g: c * idf(g, df) for g, c in rng.items()}
            s += coseno(vp, vr)
        puntajes.append(s / total)
    return sum(puntajes) / len(puntajes)


def info_mutua(pred, ref):
    """Informacion mutua entre "de que texto viene" (respuesta/referencia) y
    "que palabra es". Es una DIVERGENCIA de vocabulario: 0 cuando los dos textos
    usan las palabras igual y sube cuando difieren.

    Es informativa: mide informatividad, NO si la respuesta es correcta ni si
    esta bien escrita. Nunca se usa sola para decidir.
    """
    tp, tr = pred.lower().split(), ref.lower().split()
    if not tp or not tr:
        return 0.0
    cp, cr = Counter(tp), Counter(tr)
    n = len(tp) + len(tr)
    mi = 0.0
    for t in set(tp) | set(tr):
        pt = (cp[t] + cr[t]) / n
        for cuenta, total in ((cp[t], len(tp)), (cr[t], len(tr))):
            if cuenta:
                conjunta = cuenta / n
                mi += conjunta * math.log(conjunta / ((total / n) * pt))
    return mi


def coseno_semantico(preds, refs):
    """Similitud coseno media entre los embeddings de la respuesta y la referencia."""
    from sentence_transformers import SentenceTransformer
    modelo = SentenceTransformer(MODELO_EMBEDDINGS, device="cpu")
    ep = modelo.encode(preds, convert_to_tensor=True, normalize_embeddings=True)
    er = modelo.encode(refs, convert_to_tensor=True, normalize_embeddings=True)
    return float((ep * er).sum(dim=1).mean())


def calcular_metricas(datos, hp):
    """Calcula todas las metricas a partir del archivo de respuestas.

    `hp` son los hiperparametros de la corrida: van en las primeras columnas de
    la fila para poder ordenar y graficar el Excel por ellos (con el nombre del
    checkpoint solo no se puede).
    """
    casos = datos["casos"]
    preds = [c["respuesta"] for c in casos]
    refs = [c["referencia"] for c in casos]
    n = len(casos)

    print("\nMidiendo...")
    from bert_score import score as bert_score
    _, _, f1 = bert_score(preds, refs, lang="es", rescale_with_baseline=False, device="cpu")
    bertscore_f1 = float(f1.mean())

    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l = sum(scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds)) / n

    import sacrebleu
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score

    perdida = sum(c["perdida"] for c in casos) / n
    entropia = sum(c["entropia"] for c in casos) / n

    fuera = datos["fuera_dominio"]
    repeticiones = [repeticion_ngramas(p) for p in preds]
    razones = [razon_longitud(p, r) for p, r in zip(preds, refs)]

    return {
        "n_prueba": hp.get("num_prueba"),
        "epocas": hp.get("epocas"),
        "learning_rate": hp.get("learning_rate"),
        "rank": hp.get("rank"),
        "lora_alpha": hp.get("lora_alpha"),
        "lora_dropout": hp.get("lora_dropout"),
        "weight_decay": hp.get("weight_decay"),
        "warmup_ratio": hp.get("warmup_ratio"),
        "batch_efectivo": hp.get("batch_efectivo"),
        "seq_len": hp.get("seq_len"),
        "fecha": datos["fecha"],
        "checkpoint": datos["checkpoint"],
        "n_test": n,
        "n_fuera": len([c for c in fuera if c.get("bloque") != BLOQUE_CONTROL]),
        "exactitud_citas": round(exactitud_citas(casos), 4),
        "tasa_alucinacion": round(tasa_alucinacion(fuera), 4),
        "tasa_abstencion": tasa_abstencion(fuera),
        "bertscore_f1": round(bertscore_f1, 4),
        "coseno_sem": round(coseno_semantico(preds, refs), 4),
        "f1_tokens": round(sum(f1_tokens(p, r) for p, r in zip(preds, refs)) / n, 4),
        "rouge_l": round(rouge_l, 4),
        "bleu": round(bleu, 2),
        "cider": round(cider(preds, refs), 4),
        "perdida_test": round(perdida, 4),
        "perplejidad": round(math.exp(perdida), 2),
        "entropia": round(entropia, 4),
        "info_mutua": round(sum(info_mutua(p, r) for p, r in zip(preds, refs)) / n, 4),
        # texto degenerado
        "repeticion_4g": round(mediana(repeticiones), 4),
        "repeticion_altas": sum(1 for x in repeticiones if x > 0.30),
        "razon_longitud": round(mediana(razones), 4),
        "longitud_desviada": sum(1 for x in razones if x < 0.5 or x > 2.0),
    }


# Orden de las columnas del Excel: la linea final se pega directo en una fila.
COLUMNAS_HIPER = [
    "n_prueba", "epocas", "learning_rate", "rank", "lora_alpha",
    "lora_dropout", "weight_decay", "warmup_ratio", "batch_efectivo", "seq_len",
]
COLUMNAS_METRICAS = [
    "fecha", "checkpoint", "n_test", "n_fuera",
    "exactitud_citas", "tasa_alucinacion", "tasa_abstencion",
    "bertscore_f1", "coseno_sem", "f1_tokens",
    "rouge_l", "bleu", "cider",
    "perdida_test", "perplejidad", "entropia", "info_mutua",
    "repeticion_4g", "repeticion_altas", "razon_longitud", "longitud_desviada",
]
COLUMNAS = COLUMNAS_HIPER + COLUMNAS_METRICAS

# Valor al que apunta cada metrica, para leer los resultados de un vistazo.
OBJETIVOS = {
    "n_fuera": "fuera de dominio SIN las de control",
    "exactitud_citas": "subir, meta > 0.90",
    "tasa_alucinacion": "bajar, meta < 0.30",
    "tasa_abstencion": "subir = 1 - tasa_alucinacion",
    "bertscore_f1": "0.8-0.9 solido, > 0.9 excelente",
    "coseno_sem": "meta > 0.75",
    "perdida_test": "muy baja = memorizacion",
    "perplejidad": "bajisima = memorizacion, no calidad",
    "entropia": "informativa: baja = seguridad ciega",
    "info_mutua": "informativa, nunca sola",
    "repeticion_4g": "mediana, bajar. 0 = sin bucles",
    "repeticion_altas": "cuantas pasan del 30% repetido",
    "razon_longitud": "mediana, ideal 1.0",
    "longitud_desviada": "cuantas se desvian mas del doble",
}


def main():
    args = parse_args()
    config = load_config(args.config)

    checkpoint = args.checkpoint or config.get("model", {}).get("checkpoint_path", "")
    if not checkpoint:
        print("Falta el checkpoint: pasalo con --checkpoint o ponlo en config.yaml.")
        sys.exit(1)

    nombre = Path(checkpoint).name
    ruta_respuestas = RAIZ / "data/eval" / f"{nombre}.json"

    if ruta_respuestas.exists() and not args.regenerar:
        print(f"Reutilizando respuestas de: {ruta_respuestas}")
        print("(usa --regenerar para volver a generarlas)")
        with open(ruta_respuestas, encoding="utf-8") as f:
            datos = json.load(f)
        # Evita reutilizar una prueba hecha con --limit 10 como si fuera una corrida.
        pedido_antes = datos.get("limit", datos.get("n_test"))
        if args.limit != pedido_antes:
            print(f"\nOJO: ese archivo se genero con --limit {pedido_antes} y estas "
                  f"pidiendo {args.limit}.")
            print("Corre con --regenerar (o borralo) para no mezclar corridas.")
            sys.exit(1)
        if datos.get("muestra") != MUESTRA:
            print(f"\nOJO: ese archivo se genero con otra muestra ({datos.get('muestra')}).")
            print("Corre con --regenerar para no mezclar corridas.")
            sys.exit(1)
        # Un archivo viejo con otras preguntas fuera de dominio no es comparable.
        with open(args.fuera_dominio, encoding="utf-8") as f:
            n_fuera_ahora = len(json.load(f))
        n_fuera_antes = len(datos.get("fuera_dominio", []))
        if n_fuera_antes != n_fuera_ahora:
            print(f"\nOJO: ese archivo se genero con {n_fuera_antes} preguntas fuera "
                  f"de dominio y ahora hay {n_fuera_ahora}.")
            print("Corre con --regenerar para medir las dos corridas sobre el mismo "
                  "conjunto.")
            sys.exit(1)
        if datos["casos"] and "perdida" not in datos["casos"][0]:
            print("\nOJO: ese archivo quedo a medias (tiene las respuestas pero no la "
                  "perdida).")
            print("La corrida anterior se corto al calcular la perdida. Corre con "
                  "--regenerar.")
            sys.exit(1)
    else:
        with open(args.test, encoding="utf-8") as f:
            test = muestra_por_familia(json.load(f))
        if args.limit:
            test = test[:args.limit]
        with open(args.fuera_dominio, encoding="utf-8") as f:
            fuera = json.load(f)
        if fuera and isinstance(fuera[0], str):
            print("\nOJO: preguntas_fuera_dominio.json esta en el formato viejo "
                  "(lista de textos).")
            print('Tiene que ser: [{"pregunta": "...", "bloque": "..."}, ...]')
            sys.exit(1)
        datos = generar(ruta_respuestas, checkpoint, config, test, fuera, args.limit)

    hp = hiperparametros(checkpoint)
    resultados = calcular_metricas(datos, hp)

    faltan = [k for k in COLUMNAS_HIPER if resultados.get(k) is None]
    if faltan:
        print(f"\nOJO: no se pudo recuperar {', '.join(faltan)}.")
        print("Esas columnas van vacias en el Excel: complétalas a mano.")
        print("(los checkpoints entrenados desde ahora traen hiperparametros.json)")

    print("\n=== Hiperparametros de la corrida ===")
    for k in COLUMNAS_HIPER:
        print(f"  {k:<18} {resultados[k]}")

    print("\n=== Resultados ===")
    for k in COLUMNAS_METRICAS:
        nota = OBJETIVOS.get(k, "")
        print(f"  {k:<18} {resultados[k]}" + (f"   ({nota})" if nota else ""))

    print("\n=== Alucinacion por tipo de pregunta ===")
    for bloque, tasa in alucinacion_por_bloque(datos["fuera_dominio"]).items():
        print(f"  {bloque:<24} {tasa}")

    resultados["por_familia"] = por_familia(datos["casos"])
    print("\n=== Por familia del test (dice_no_saber: alto en negativos, bajo en el resto) ===")
    for familia, valores in resultados["por_familia"].items():
        print(f"  {familia:<10} n={valores['n']:<4} exactitud_citas={valores['exactitud_citas']}"
              f"  dice_no_saber={valores['dice_no_saber']}")

    # Solo en los archivos de t1 y t2: se revisan a ojo.
    control = [c for c in datos["fuera_dominio"]
               if c.get("bloque") == BLOQUE_CONTROL]
    if control:
        print(f"\n=== Control de conocimiento general ({len(control)}) "
              f"-- revisar a mano ===")
        for c in control:
            respuesta = " ".join(c["respuesta"].split())[:90]
            print(f"  {c['pregunta']}")
            print(f"    -> {respuesta}...")

    print("\n=== Linea para el Excel (pegar en una fila) ===")
    print("\t".join(COLUMNAS))
    print("\t".join(str(resultados[k]) for k in COLUMNAS))

    ruta_metricas = RAIZ / "data/eval" / f"{nombre}_metricas.json"
    with open(ruta_metricas, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"\nRespaldo de las metricas: {ruta_metricas}")


if __name__ == "__main__":
    main()
