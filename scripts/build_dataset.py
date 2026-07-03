"""Genera el dataset de fine-tuning desde los textos extraídos.

Uso:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --mode heuristic --format alpaca
    python scripts/build_dataset.py --mode template    # plantillas para rellenar manualmente
    python scripts/build_dataset.py --input-dir data/extracted
    python scripts/build_dataset.py --types RESOLUCION_ACADEMICA,RESOLUCION_SUPERIOR

Modos:
    heuristic  (por defecto) — extrae pares QA automáticamente del texto normativo
    template   — genera entradas vacías para completar manualmente

Lee archivos Markdown (.md) de forma recursiva desde data/extracted/. Cada .md lleva
frontmatter YAML (tipo, titulo, numero, anio) generado por extract_text.py; se usa ese
'titulo' como nombre del documento para que qa_builder construya citas correctas.

El dataset se guarda en data/dataset/dataset_alpaca.json (o sharegpt).
Este archivo es el que se usa para fine-tuning (Colab o Local dependiendo de los recursos).
"""
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
from src.config_loader import load_config
from src.dataset.qa_builder import (
    generate_heuristic, generate_template, save_dataset, QAPair
)


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Separa el frontmatter YAML del cuerpo de un .md.

    Retorna (metadata, cuerpo). Si no hay frontmatter, retorna ({}, texto).
    Parser simple key: value (suficiente para el frontmatter plano que generamos).
    """
    if not text.startswith("---"):
        return {}, text
    # Buscar el cierre del frontmatter (segundo '---' en su propia línea)
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    raw_meta, body = m.group(1), m.group(2)
    meta = {}
    for line in raw_meta.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            val = val.strip().strip('"')
            meta[key.strip()] = val
    return meta, body.lstrip("\n")


def main():
    parser = argparse.ArgumentParser(description="Genera dataset de fine-tuning.")
    parser.add_argument("--input-dir", default=None, help="Directorio con archivos .md (recursivo)")
    parser.add_argument("--output-dir", default=None, help="Directorio de salida")
    parser.add_argument("--mode", default="heuristic", choices=["heuristic", "template"])
    parser.add_argument("--format", default=None, choices=["alpaca", "sharegpt"])
    parser.add_argument("--types", default=None,
                        help="Lista separada por comas de tipos a incluir (ej. RESOLUCION_ACADEMICA)")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    types_filter = {t.strip() for t in args.types.split(",")} if args.types else None

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

    # Leer Markdown (.md) recursivamente. La extracción oficial (extract_text.py)
    # genera .md con frontmatter YAML; es el único formato de entrada soportado.
    src_files = sorted(input_dir.rglob("*.md"))
    if not src_files:
        print(f"No se encontraron archivos .md en: {input_dir}")
        print("Ejecuta primero: python scripts/extract_text.py")
        return

    print(f"Modo: {args.mode} | Formato: {fmt} | Fuente: Markdown")
    if types_filter:
        print(f"Filtro tipos: {', '.join(sorted(types_filter))}")
    print(f"Procesando {len(src_files)} documento(s)...\n")

    all_pairs: list[QAPair] = []
    by_type: dict[str, int] = {}
    omitidos = 0

    for src_path in src_files:
        raw = src_path.read_text(encoding="utf-8")

        meta, body = _split_frontmatter(raw)
        tipo = meta.get("tipo", "")

        if types_filter and tipo not in types_filter:
            omitidos += 1
            continue

        # qa_builder lee el Markdown con sus encabezados intactos + el frontmatter;
        # por eso NO se aplana con clean() (destruiría la estructura ##/###).
        if args.mode == "heuristic":
            pairs = generate_heuristic(body, meta)
        else:
            pairs = generate_template(meta)

        rel = src_path.relative_to(input_dir).as_posix()
        print(f"  [{tipo or '—'}] {rel}: {len(pairs)} pares QA")
        by_type[tipo] = by_type.get(tipo, 0) + len(pairs)
        all_pairs.extend(pairs)

    # Eliminar preguntas repetidas (el corpus trae archivos duplicados, p.ej.
    # "3_2015.pdf" y "3_2015_9JYqTCE.pdf" → misma cita): se conserva la primera
    # para no dejar la misma pregunta con dos respuestas distintas.
    seen_q: set[str] = set()
    unique_pairs: list[QAPair] = []
    for p in all_pairs:
        if p.question in seen_q:
            continue
        seen_q.add(p.question)
        unique_pairs.append(p)
    duplicados = len(all_pairs) - len(unique_pairs)
    all_pairs = unique_pairs

    output_file = output_dir / f"dataset_{fmt}.json"
    save_dataset(all_pairs, output_file, fmt=fmt)
    if duplicados:
        print(f"  ({duplicados} pares duplicados eliminados)")

    if by_type:
        print("\nPares QA por tipo de documento:")
        for tipo in sorted(by_type, key=lambda t: -by_type[t]):
            print(f"  {by_type[tipo]:5d}  {tipo or '—'}")
    if omitidos:
        print(f"\nDocumentos omitidos por filtro de tipo: {omitidos}")

    print(f"\nTotal: {len(all_pairs)} pares QA en {output_file}")

    if args.mode == "template":
        print("\nATENCION: las plantillas requieren completarse manualmente.")
        print(f"Edita {output_file} y reemplaza los campos [RESPUESTA ...] con información real.")
        print("Cuantas más respuestas reales y precisas agregues, mejor será el modelo fine-tuneado.")


if __name__ == "__main__":
    main()
