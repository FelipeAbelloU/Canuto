"""Genera el dataset de entrenamiento a partir de los .md extraidos.

Lee los .md de data/extracted/ (con su frontmatter) y arma los pares
pregunta-respuesta en data/dataset/dataset_alpaca.json. Ademas de los pares
normales agrega ejemplos negativos (preguntas cuya respuesta es "no tengo
informacion"), para que el modelo aprenda a reconocer lo que no sabe.

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
    cita_key, cita_completa, ultimo_articulo, save_dataset, enmascarar, QAPair
)

# Semilla fija: los numeros falsos de los negativos salen siempre iguales.
SEMILLA = 42


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
    """Lee config/correcciones_citas.tsv: ruta del PDF -> fila (accion, numero, anio).

    Son los PDF que tienen en el nombre un numero distinto al impreso, revisados a
    mano. No se renombra ningun PDF: la correccion se aplica aqui, asi un OCR nuevo
    no la deshace. accion = "citar_como" (usar numero y anio) o "excluir" (duplicado).
    """
    if not ruta.exists():
        return {}
    with open(ruta, encoding="utf-8") as f:
        return {fila["archivo"]: fila for fila in csv.DictReader(f, delimiter="\t")}


def cargar_documentos(archivos: list[Path], correcciones: dict) -> tuple[list, set, int]:
    """Lee los .md y les aplica las correcciones de citas. Lo usan este script y
    generar_preguntas_tema.py, para que los dos vean los mismos documentos.

    Devuelve ([(ruta, meta, cuerpo)], filas de correcciones usadas, excluidos).
    """
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


def cargar_clave(ruta: Path) -> dict:
    """Lee config/documentos_clave.txt: (tipo, numero, año) -> nombre comun."""
    clave = {}
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.split("#")[0]
        if "|" in linea:
            cita, nombre = linea.split("|")
            clave[tuple(cita.split())] = nombre.strip()
    return clave


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

    # Primera pasada: indice de todos los documentos del corpus con su ultimo
    # articulo. Con esto los negativos preguntan por cosas que de verdad no
    # estan en el corpus. Si un documento esta repetido se queda el mayor.
    # Antes de indexar se corrigen las citas de los PDF mal nombrados.
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

    # Documentos clave (tercera redaccion) y articulos modificados (nota en la
    # respuesta). Se avisa si alguna cita de las listas no esta en el corpus.
    citas = {cita_key(meta): cita_completa(meta) for _, meta, _ in docs}
    clave = cargar_clave(root / "config/documentos_clave.txt")
    modificados = {doc: {art: citas.get(por) for art, por in arts.items()}
                   for doc, arts in cargar_modificados(root / "config/articulos_modificados.tsv").items()}
    faltan = [d for d in clave if d not in citas] + [d for d in modificados if d not in citas]
    faltan += [(doc, art) for doc, arts in modificados.items() for art, cita in arts.items() if cita is None]
    print(f"Documentos clave: {len(clave)} · documentos con articulos modificados: {len(modificados)}")
    for d in faltan:
        print(f"  OJO: no se encontro en el corpus: {d}")
    print()

    rng = random.Random(SEMILLA)
    all_pairs: list[QAPair] = []
    negativos: list[QAPair] = []
    by_type: dict[str, int] = {}
    omitidos = 0

    for src_path, meta, body in docs:
        tipo = meta.get("tipo", "")

        if types_filter and tipo not in types_filter:
            omitidos += 1
            continue

        key = cita_key(meta)
        pairs = generate_heuristic(body, meta, clave.get(key, ""), modificados.get(key))
        negs = generate_negatives(meta, indice, rng)

        rel = src_path.relative_to(input_dir).as_posix()
        print(f"  [{tipo or '—'}] {rel}: {len(pairs)} pares QA + {len(negs)} negativos")
        by_type[tipo] = by_type.get(tipo, 0) + len(pairs)
        all_pairs.extend(pairs)
        negativos.extend(negs)

    # Preguntas abiertas de los documentos clave: B las escribio Qwen
    # (generar_preguntas_tema.py) y E son las frecuentes, escritas a mano. Cada
    # una lleva la pregunta con cita de su articulo ("cita"), y con ella se toma
    # la MISMA respuesta de la familia A.
    respuesta_de = {p.question: p.answer for p in all_pairs}
    for archivo, familia in [("preguntas_tema.json", "B"), ("preguntas_frecuentes.json", "E")]:
        ruta = root / "data/dataset" / archivo
        if not ruta.exists():
            continue
        with open(ruta, encoding="utf-8") as f:
            abiertas = [p for p in json.load(f) if p.get("filtro", "ok") == "ok"]
        sin_articulo = [p for p in abiertas if p["cita"] not in respuesta_de]
        all_pairs.extend(QAPair(p["pregunta"], respuesta_de[p["cita"]], familia)
                         for p in abiertas if p["cita"] in respuesta_de)
        print(f"Preguntas abiertas (familia {familia}): {len(abiertas) - len(sin_articulo)}")
        for p in sin_articulo:
            print(f"  OJO: no se encontro el articulo de: {p['pregunta']}")

    # Negativos que no salen de ningun documento (otras universidades, datos
    # privados, cultura general...).
    with open(root / "data/dataset/preguntas_generales.json", encoding="utf-8") as f:
        generales = json.load(f)
    negativos.extend(generate_generales(rng, generales))

    # Ninguna pregunta del examen fuera de dominio puede quedar en el dataset:
    # si el modelo la ve al entrenar, la tasa de alucinacion sale mejor de lo que es.
    with open(root / "data/dataset/preguntas_fuera_dominio.json", encoding="utf-8") as f:
        examen = {p["pregunta"] for p in json.load(f)}
    negativos = [p for p in negativos if p.question not in examen]
    all_pairs = [p for p in all_pairs if p.question not in examen]
    n_negativos = len(negativos)
    all_pairs.extend(negativos)

    # El corpus trae documentos duplicados, asi que se quitan las preguntas repetidas.
    seen_q: set[str] = set()
    unique_pairs: list[QAPair] = []
    for p in all_pairs:
        if p.question in seen_q:
            continue
        seen_q.add(p.question)
        unique_pairs.append(p)
    duplicados = len(all_pairs) - len(unique_pairs)
    all_pairs = unique_pairs

    # Cedulas, pasaportes y codigos de estudiante -> "(dato reservado)". Varias
    # preguntas comparten la misma respuesta, asi que cada texto se enmascara
    # una sola vez (era el 70 % del tiempo del script).
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
    print(f"Pares con datos personales enmascarados: {enmascarados}")
    print(f"Total: {len(all_pairs)} pares QA en {output_file}")


if __name__ == "__main__":
    main()
