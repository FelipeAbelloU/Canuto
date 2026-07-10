"""Evalua el modelo fine-tuneado sobre el conjunto de prueba (data/dataset/test.json).

Calcula el "core" de metricas de la tesis:
  - BERTScore (F1): parecido semantico entre la respuesta generada y la esperada.
  - ROUGE-L / BLEU: solapamiento de texto (metricas de apoyo).
  - Exactitud de citas: si la respuesta menciona el documento correcto (No. X de AAAA).
  - Tasa de alucinacion: sobre preguntas fuera de dominio, cuantas veces el modelo
    inventa en vez de reconocer que no tiene esa informacion.

El modelo a evaluar es el que este configurado en config/config.yaml (checkpoint_path).
Mas adelante se pueden agregar mas metricas segun lo pida la asesora.

Uso:
    python scripts/evaluate.py
    python scripts/evaluate.py --test data/dataset/test.json --limit 50
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.factory import create_pipeline

# Frases con las que el modelo reconoce que no tiene la informacion (no alucina)
_FRASES_NO_SE = [
    "no tengo", "no dispongo", "no cuento con", "no encuentro", "no aparece",
    "no está en", "no se encuentra", "no hay información", "no tengo información",
    "no tengo certeza", "no forma parte", "no puedo", "no está disponible",
    "no corresponde", "fuera de mi", "no dispongo de",
]

# Cita de un documento: "No. 10 de 2015", "N° 007 de 1990", "10 de 2015"
_CITA_RE = re.compile(r"N[°ºo\.]*\s*0*(\d{1,4})\s+de\s+(\d{4})", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluacion del modelo con el core de metricas")
    p.add_argument("--test", default=str(ROOT / "data/dataset/test.json"))
    p.add_argument("--fuera-dominio", default=str(ROOT / "data/dataset/preguntas_fuera_dominio.json"))
    p.add_argument("--config", default=str(ROOT / "config/config.yaml"))
    p.add_argument("--out", default=str(ROOT / "data/dataset/resultados_evaluacion.json"))
    p.add_argument("--limit", type=int, default=0, help="Evaluar solo N ejemplos (0 = todos)")
    return p.parse_args()


def citas(texto):
    """Conjunto de citas (numero, anio) encontradas en un texto."""
    return {(n.lstrip("0") or "0", a) for n, a in _CITA_RE.findall(texto)}


def reconoce_no_saber(texto):
    t = texto.lower()
    return any(f in t for f in _FRASES_NO_SE)


def generar_respuestas(pipeline, preguntas):
    """Pasa cada pregunta por el modelo (sin memoria entre preguntas)."""
    respuestas = []
    for i, q in enumerate(preguntas, 1):
        pipeline.reset_history()
        respuestas.append(pipeline.query(q)["answer"])
        print(f"  {i}/{len(preguntas)}", end="\r")
    print()
    return respuestas


def main():
    args = parse_args()

    pipeline = create_pipeline(args.config)
    if pipeline.model is None or not pipeline.model.is_available():
        print("No hay checkpoint configurado en config.yaml (model.checkpoint_path).")
        sys.exit(1)

    with open(args.test, encoding="utf-8") as f:
        data = json.load(f)
    if args.limit:
        data = data[:args.limit]

    preguntas = [r["instruction"] for r in data]
    esperadas = [r["output"] for r in data]

    print(f"Generando respuestas para {len(preguntas)} preguntas de prueba...")
    predichas = generar_respuestas(pipeline, preguntas)

    # --- BERTScore (parecido semantico) ---
    from bert_score import score as bert_score
    _, _, f1 = bert_score(predichas, esperadas, lang="es", rescale_with_baseline=False)
    bertscore_f1 = float(f1.mean())

    # --- ROUGE-L ---
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l = sum(scorer.score(ref, pred)["rougeL"].fmeasure
                  for ref, pred in zip(esperadas, predichas)) / len(data)

    # --- BLEU ---
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(predichas, [esperadas]).score

    # --- Exactitud de citas ---
    con_cita = aciertos = 0
    for ref, pred in zip(esperadas, predichas):
        objetivo = citas(ref)
        if objetivo:
            con_cita += 1
            if objetivo & citas(pred):
                aciertos += 1
    exactitud_citas = (aciertos / con_cita) if con_cita else 0.0

    # --- Tasa de alucinacion (preguntas fuera de dominio) ---
    with open(args.fuera_dominio, encoding="utf-8") as f:
        fuera = json.load(f)
    print(f"Generando respuestas para {len(fuera)} preguntas fuera de dominio...")
    resp_fuera = generar_respuestas(pipeline, fuera)
    alucina = sum(0 if reconoce_no_saber(r) else 1 for r in resp_fuera)
    tasa_alucinacion = alucina / len(fuera)

    resultados = {
        "modelo": pipeline.model.name,
        "n_prueba": len(data),
        "bertscore_f1": round(bertscore_f1, 4),
        "rouge_l": round(rouge_l, 4),
        "bleu": round(bleu, 2),
        "exactitud_citas": round(exactitud_citas, 4),
        "citas_evaluadas": con_cita,
        "tasa_alucinacion": round(tasa_alucinacion, 4),
    }

    print("\n=== Resultados ===")
    for k, v in resultados.items():
        print(f"  {k}: {v}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"\nGuardado en: {args.out}")


if __name__ == "__main__":
    main()
