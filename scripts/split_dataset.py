"""Divide el dataset en entrenamiento (70%) y prueba (30%).

El 70% se usa para entrenar el modelo y el 30% queda apartado para evaluar
(scripts/evaluate.py). La division es aleatoria pero con semilla fija, para que
sea siempre la misma y los experimentos sean comparables.

Uso:
    python scripts/split_dataset.py
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="Split 70/30 del dataset")
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

    # Mezclar con semilla fija y cortar 70/30
    random.seed(args.seed)
    random.shuffle(data)
    n_test = int(len(data) * args.test_frac)
    test = data[:n_test]
    train = data[n_test:]

    for path, part in [(args.train_out, train), (args.test_out, test)]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(part, f, ensure_ascii=False, indent=2)

    print(f"Total: {len(data)} pares")
    print(f"  Entrenamiento: {len(train)} -> {args.train_out}")
    print(f"  Prueba:        {len(test)} -> {args.test_out}")


if __name__ == "__main__":
    main()
