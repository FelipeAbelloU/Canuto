"""Arma data/dataset/dataset_alpaca.json desde los .md de data/extracted/:
pares con cita (A), preguntas abiertas (B y E) y negativos.

Uso:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --types RESOLUCION_ACADEMICA,RESOLUCION_SUPERIOR
"""
import sys
import re
import csv
import json
import random
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.config_loader import load_config
from src.dataset.qa_builder import (
    generate_heuristic, generate_negatives, generate_generales,
    cita_key, cita_completa, ultimo_articulo, save_dataset, enmascarar, tiene_basura_ocr, QAPair
)

# Fija para que los numeros falsos de los negativos salgan siempre iguales.
SEMILLA = 42
# Solo entran las que aprobo la revision; en una muestra calificada a mano
# quedo 1 mala de 34.
USAR_PREGUNTAS_QWEN = True


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Separa el frontmatter YAML del cuerpo del .md. Devuelve (metadata, cuerpo)."""
    if not text.startswith("---"):
        return {}, text
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


def cargar_correcciones(ruta: Path) -> dict:
    """Ruta del PDF -> fila. Se corrige aqui y no renombrando el PDF, para que un
    OCR nuevo no deshaga la correccion. accion: "citar_como" o "excluir"."""
    if not ruta.exists():
        return {}
    with open(ruta, encoding="utf-8") as f:
        return {fila["archivo"]: fila for fila in csv.DictReader(f, delimiter="\t")}


def cargar_documentos(archivos: list[Path], correcciones: dict) -> tuple[list, set, int]:
    """Lee los .md con las citas corregidas. Lo usa tambien generar_preguntas_tema.py.
    Devuelve ([(ruta, meta, cuerpo)], filas de correcciones usadas, excluidos)."""
    docs, usadas, excluidos = [], set(), 0
    for ruta in archivos:
        meta, body = _split_frontmatter(ruta.read_text(encoding="utf-8"))
        fila = correcciones.get(meta.get("ruta_origen", ""))
        if fila:
            usadas.add(fila["archivo"])
            if fila["accion"] == "excluir":
                excluidos += 1
                continue
            meta["numero"], meta["anio"] = fila["numero"], fila["anio"]
        docs.append((ruta, meta, body))
    return docs, usadas, excluidos


def cargar_modificados(ruta: Path) -> dict:
    """Lee config/articulos_modificados.tsv: (tipo, numero, año) -> {articulo: (tipo, numero, año)}."""
    modificados: dict = {}
    with open(ruta, encoding="utf-8") as f:
        for fila in csv.DictReader(f, delimiter="\t"):
            doc = tuple(fila["documento"].split())
            modificados.setdefault(doc, {})[fila["articulo"]] = tuple(fila["modificado_por"].split())
    return modificados


def main():
    parser = argparse.ArgumentParser(description="Genera dataset de fine-tuning.")
    parser.add_argument("--input-dir", default=None, help="Directorio con archivos .md (recursivo)")
    parser.add_argument("--output-dir", default=None, help="Directorio de salida")
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

    if not input_dir.is_absolute():
        input_dir = (root / input_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted(input_dir.rglob("*.md"))
    if not src_files:
        print(f"No se encontraron archivos .md en: {input_dir}")
        print("Ejecuta primero: python scripts/extract_text.py")
        return

    if types_filter:
        print(f"Filtro tipos: {', '.join(sorted(types_filter))}")
    print(f"Procesando {len(src_files)} documento(s)...\n")

    # Indice del corpus (cita -> ultimo articulo): los negativos preguntan por
    # lo que de verdad no existe.
    correcciones = cargar_correcciones(root / "config/correcciones_citas.tsv")
    docs, usadas, excluidos = cargar_documentos(src_files, correcciones)
    indice: dict = {}
    for _, meta, body in docs:
        key = cita_key(meta)
        indice[key] = max(indice.get(key, 0), ultimo_articulo(body))
    print(f"Correcciones de citas: {len(usadas)} de {len(correcciones)} aplicadas "
          f"({excluidos} duplicados excluidos)")
    for archivo in sorted(set(correcciones) - usadas):
        print(f"  OJO: no se encontro el .md de {archivo}")
    print(f"Indice del corpus: {len(indice)} documentos distintos")

    citas = {cita_key(meta): cita_completa(meta) for _, meta, _ in docs}
    modificados = {doc: {art: citas.get(por) for art, por in arts.items()}
                   for doc, arts in cargar_modificados(root / "config/articulos_modificados.tsv").items()}
    faltan = [d for d in modificados if d not in citas]
    faltan += [(doc, art) for doc, arts in modificados.items() for art, cita in arts.items() if cita is None]
    print(f"Documentos con articulos modificados: {len(modificados)}")
    for d in faltan:
        print(f"  OJO: no se encontro en el corpus: {d}")
    print()

    rng = random.Random(SEMILLA)
    all_pairs: list[QAPair] = []
    negativos: list[QAPair] = []
    preguntas_negativas: set[str] = set()
    by_type: dict[str, int] = {}
    omitidos = 0
    quitados_ocr: set[str] = set()

    for src_path, meta, body in docs:
        tipo = meta.get("tipo", "")

        if types_filter and tipo not in types_filter:
            omitidos += 1
            continue

        key = cita_key(meta)
        pairs = generate_heuristic(body, meta, modificados.get(key))
        # Se quitan antes de contar los negativos para que sigan 1:1.
        for p in pairs:
            if tiene_basura_ocr(p.answer):
                quitados_ocr.add(p.question)
        pairs = [p for p in pairs if p.question not in quitados_ocr]
        negs = generate_negatives(meta, indice, rng, len(pairs), preguntas_negativas)   # 1:1 por documento

        rel = src_path.relative_to(input_dir).as_posix()
        print(f"  [{tipo or '—'}] {rel}: {len(pairs)} pares QA + {len(negs)} negativos")
        by_type[tipo] = by_type.get(tipo, 0) + len(pairs)
        all_pairs.extend(pairs)
        negativos.extend(negs)

    # B (Qwen) y E (a mano) toman la respuesta A de su articulo por "cita".
    respuesta_de = {p.question: p.answer for p in all_pairs}
    abiertas_por_familia = [("preguntas_frecuentes.json", "E")]
    if USAR_PREGUNTAS_QWEN:
        abiertas_por_familia.append(("preguntas_tema.json", "B"))
    for archivo, familia in abiertas_por_familia:
        ruta = root / "data/dataset" / archivo
        if not ruta.exists():
            continue
        with open(ruta, encoding="utf-8") as f:
            abiertas = [p for p in json.load(f) if p.get("filtro", "ok") == "ok"]
        por_ocr = [p for p in abiertas if p["cita"] in quitados_ocr]
        sin_articulo = [p for p in abiertas if p["cita"] not in respuesta_de and p["cita"] not in quitados_ocr]
        all_pairs.extend(QAPair(p["pregunta"], respuesta_de[p["cita"]], familia)
                         for p in abiertas if p["cita"] in respuesta_de)
        print(f"Preguntas abiertas (familia {familia}): "
              f"{len(abiertas) - len(sin_articulo) - len(por_ocr)} ({len(por_ocr)} quitadas por el OCR)")
        for p in sin_articulo:
            print(f"  OJO: no se encontro el articulo de: {p['pregunta']}")

    with open(root / "data/dataset/preguntas_generales.json", encoding="utf-8") as f:
        generales = json.load(f)
    negativos.extend(generate_generales(rng, generales))

    # Si el modelo viera el examen fuera de dominio, la alucinacion saldria mejor.
    with open(root / "data/dataset/preguntas_fuera_dominio.json", encoding="utf-8") as f:
        examen = {p["pregunta"] for p in json.load(f)}
    negativos = [p for p in negativos if p.question not in examen]
    all_pairs = [p for p in all_pairs if p.question not in examen]
    n_negativos = len(negativos)
    all_pairs.extend(negativos)

    # El corpus trae documentos duplicados.
    seen_q: set[str] = set()
    unique_pairs: list[QAPair] = []
    for p in all_pairs:
        if p.question in seen_q:
            continue
        seen_q.add(p.question)
        unique_pairs.append(p)
    duplicados = len(all_pairs) - len(unique_pairs)
    all_pairs = unique_pairs

    # Cada respuesta se repite en 2 o 3 pares: se enmascara una sola vez.
    enmascarados = 0
    enmascarar_una_vez = lru_cache(maxsize=None)(enmascarar)
    for p in all_pairs:
        q, a = enmascarar_una_vez(p.question), enmascarar_una_vez(p.answer)
        if (q, a) != (p.question, p.answer):
            enmascarados += 1
        p.question, p.answer = q, a

    output_file = output_dir / "dataset_alpaca.json"
    save_dataset(all_pairs, output_file)
    if duplicados:
        print(f"  ({duplicados} pares duplicados eliminados)")

    if by_type:
        print("\nPares QA por tipo de documento:")
        for tipo in sorted(by_type, key=lambda t: -by_type[t]):
            print(f"  {by_type[tipo]:5d}  {tipo or '—'}")
    if omitidos:
        print(f"\nDocumentos omitidos por filtro de tipo: {omitidos}")

    print(f"\nNegativos: {n_negativos} (antes de quitar duplicados)")
    print(f"Pares A quitados por basura del OCR: {len(quitados_ocr)}")
    print(f"Pares con datos personales enmascarados: {enmascarados}")
    print(f"Total: {len(all_pairs)} pares QA en {output_file}")


if __name__ == "__main__":
    main()
