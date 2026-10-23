"""Divide el dataset 70/30 por redaccion, no par por par.

Sin RAG, un articulo que cae entero en la prueba es una pregunta imposible. Las
redacciones de un articulo comparten respuesta: de cada grupo de 2 o mas va UNA
a la prueba y el resto a entrenamiento. Los negativos se reparten al azar.

Uso:
    python scripts/split_dataset.py
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="Split 70/30 del dataset por redaccion")
    p.add_argument("--dataset", default=str(ROOT / "data/dataset/dataset_alpaca.json"))
    p.add_argument("--train-out", default=str(ROOT / "data/dataset/train.json"))
    p.add_argument("--test-out", default=str(ROOT / "data/dataset/test.json"))
    p.add_argument("--test-frac", type=float, default=0.30, help="Fraccion para prueba")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(args.seed)
    positivos = [r for r in data if r.get("familia", "A") != "negativo"]
    negativos = [r for r in data if r.get("familia", "A") == "negativo"]

    # Un grupo = un articulo o una parte de articulo.
    grupos: dict[str, list] = {}
    for r in positivos:
        grupos.setdefault(r["output"], []).append(r)
    grupos = list(grupos.values())
    rng.shuffle(grupos)

    meta_pos = int(len(positivos) * args.test_frac)
    train, test = [], []
    for grupo in grupos:
        if len(test) < meta_pos and len(grupo) >= 2:
            # Mejor una abierta (B o E): asi pregunta un estudiante real.
            abiertas = [i for i, r in enumerate(grupo) if r.get("familia") in ("B", "E")]
            elegida = rng.choice(abiertas) if abiertas else rng.randrange(len(grupo))
            test.append(grupo.pop(elegida))
        train.extend(grupo)
    test_pos = len(test)

    rng.shuffle(negativos)
    n_test_neg = int(len(negativos) * args.test_frac)
    test.extend(negativos[:n_test_neg])
    train.extend(negativos[n_test_neg:])

    # Sin estas dos condiciones la nota de la prueba no vale.
    preguntas_train = {r["instruction"] for r in train}
    respuestas_train = {r["output"] for r in train}
    repetidas = [r for r in test if r["instruction"] in preguntas_train]
    sin_hermana = [r for r in test[:test_pos] if r["output"] not in respuestas_train]
    if repetidas or sin_hermana:
        print(f"ERROR: {len(repetidas)} preguntas de la prueba estan tambien en "
              f"entrenamiento y {len(sin_hermana)} articulos de la prueba no tienen "
              "otra redaccion en entrenamiento. No se guarda nada.")
        sys.exit(1)

    rng.shuffle(train)
    rng.shuffle(test)
    for path, part in [(args.train_out, train), (args.test_out, test)]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(part, f, ensure_ascii=False, indent=2)

    print(f"Total: {len(data)} pares ({len(positivos)} positivos, {len(negativos)} negativos)")
    print(f"  Entrenamiento: {len(train)} -> {args.train_out}")
    print(f"  Prueba:        {len(test)} ({len(test) / len(data):.1%}) -> {args.test_out}")
    print(f"  Positivos en prueba: {test_pos} · negativos en prueba: {n_test_neg}")
    print("  Sin fugas: cada articulo de la prueba tiene otra redaccion en entrenamiento")
    if test_pos < meta_pos:
        print(f"  OJO: solo {test_pos} positivos en prueba de {meta_pos} buscados: no hay "
              "suficientes articulos con 2 o mas redacciones (falta F4).")


if __name__ == "__main__":
    main()
