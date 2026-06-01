"""Genera el dataset de fine-tuning desde los textos extraídos.

Uso:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --mode heuristic --format alpaca
    python scripts/build_dataset.py --mode template    # plantillas para rellenar manualmente

Modos:
    heuristic  (por defecto) — extrae pares QA automáticamente del texto normativo
    template   — genera entradas vacías para completar manualmente

El dataset se guarda en data/dataset/dataset_alpaca.json (o sharegpt).
Este archivo es el que se usa para fine-tuning.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
from src.config_loader import load_config
from src.dataset.cleaner import clean
from src.dataset.qa_builder import (
    generate_heuristic, generate_template, save_dataset, QAPair
)


def main():
    parser = argparse.ArgumentParser(description="Genera dataset de fine-tuning.")
    parser.add_argument("--input-dir", default=None, help="Directorio con archivos .txt")
    parser.add_argument("--output-dir", default=None, help="Directorio de salida")
    parser.add_argument("--mode", default="heuristic", choices=["heuristic", "template"])
    parser.add_argument("--format", default=None, choices=["alpaca", "sharegpt"])
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ds_cfg = config.get("dataset", {})

    root = Path(__file__).parent.parent
    input_dir = Path(args.input_dir or ds_cfg.get("input_dir", "data/extracted"))
    output_dir = Path(args.output_dir or ds_cfg.get("output_dir", "data/dataset"))
    fmt = args.format or ds_cfg.get("format", "alpaca")

    if not input_dir.is_absolute():
        input_dir = (root / input_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No se encontraron archivos .txt en: {input_dir}")
        print("Ejecuta primero: python scripts/extract_text.py")
        return

    print(f"Modo: {args.mode} | Formato: {fmt}")
    print(f"Procesando {len(txt_files)} texto(s)...\n")

    all_pairs: list[QAPair] = []

    for txt_path in txt_files:
        text = txt_path.read_text(encoding="utf-8")
        text = clean(text)

        if args.mode == "heuristic":
            pairs = generate_heuristic(text, txt_path.stem)
        else:
            pairs = generate_template(txt_path.stem)

        print(f"  {txt_path.name}: {len(pairs)} pares QA")
        all_pairs.extend(pairs)

    output_file = output_dir / f"dataset_{fmt}.json"
    save_dataset(all_pairs, output_file, fmt=fmt)

    print(f"\nTotal: {len(all_pairs)} pares QA en {output_file}")

    if args.mode == "template":
        print("\nATENCION: las plantillas requieren completarse manualmente.")
        print(f"Edita {output_file} y reemplaza los campos [RESPUESTA ...] con información real.")
        print("Cuantas más respuestas reales y precisas agregues, mejor será el modelo fine-tuneado.")


if __name__ == "__main__":
    main()
