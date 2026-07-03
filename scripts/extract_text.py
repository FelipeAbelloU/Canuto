"""Extrae texto de los PDFs normativos y los guarda como Markdown (.md) estructurado.

El corpus sigue la estructura de SIRIUS:
    PDF/<año>/si/normatividad/<TIPO>/<numero>_<año>.pdf

La salida espeja esa estructura en data/extracted, en formato Markdown:
    data/extracted/<año>/<TIPO>/<numero>_<año>.md

Cada .md lleva frontmatter YAML con metadata (tipo, año, número, fuente) derivada
de la ruta, y el cuerpo con la estructura legal promovida a encabezados Markdown
(CONSIDERANDO/RESUELVE -> ##, ARTÍCULO -> ###, PARÁGRAFO -> ####).

Uso:
    # Solo PDFs digitales (rápido). Los escaneados se cuentan pero se omiten:
    python scripts/extract_text.py

    # Filtrar por tipo y/o año (para entrenar por subconjuntos):
    python scripts/extract_text.py --types RESOLUCION_ACADEMICA,RESOLUCION_SUPERIOR
    python scripts/extract_text.py --years 2024,2025,2026

    # Probar con pocos documentos:
    python scripts/extract_text.py --limit 10

    # Incluir OCR para escaneados (LENTO en CPU; usar en GPU/workstation):
    python scripts/extract_text.py --ocr

Los archivos ya procesados se omiten (borrar el .md para reprocesar).
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Forzar UTF-8 en stdout (Windows muestra � en consola aunque el texto sea correcto)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config_loader import load_config
from src.extraction.pdf_digital import extract as extract_digital, is_likely_digital
from src.extraction.pdf_scanned import extract as extract_scanned
from src.extraction.to_markdown import parse_meta_from_path, to_markdown


def _parse_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {v.strip() for v in value.split(",") if v.strip()}


def main():
    parser = argparse.ArgumentParser(description="Extrae texto de PDFs normativos a Markdown.")
    parser.add_argument("--pdf-dir", default=None, help="Directorio raíz del corpus de PDFs")
    parser.add_argument("--output-dir", default=None, help="Directorio de salida para .md")
    parser.add_argument("--ocr", action="store_true",
                        help="Procesar también PDFs escaneados/corruptos con OCR docling (usar en la workstation con GPU)")
    parser.add_argument("--types", default=None,
                        help="Lista separada por comas de tipos a incluir (ej. RESOLUCION_ACADEMICA)")
    parser.add_argument("--years", default=None,
                        help="Lista separada por comas de años a incluir (ej. 2024,2025,2026)")
    parser.add_argument("--limit", type=int, default=None, help="Procesar a lo sumo N PDFs (pruebas)")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ext_cfg = config.get("extraction", {})

    pdf_dir = Path(args.pdf_dir or ext_cfg.get("pdf_dir", "PDF"))
    output_dir = Path(args.output_dir or ext_cfg.get("output_dir", "data/extracted"))

    root = Path(__file__).parent.parent
    if not pdf_dir.is_absolute():
        pdf_dir = (root / pdf_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()

    types_filter = _parse_csv(args.types)
    years_filter = _parse_csv(args.years)

    # Recorrer el corpus recursivamente respetando su estructura de carpetas.
    pdfs = sorted(p for p in pdf_dir.rglob("*.pdf"))
    if not pdfs:
        print(f"No se encontraron PDFs en: {pdf_dir}")
        return

    print(f"Corpus: {pdf_dir}")
    print(f"Salida: {output_dir}")
    print(f"OCR escaneados: {'SÍ (docling)' if args.ocr else 'NO (se omiten)'}")
    if types_filter:
        print(f"Filtro tipos: {', '.join(sorted(types_filter))}")
    if years_filter:
        print(f"Filtro años: {', '.join(sorted(years_filter))}")
    print()

    stats = {"digital": 0, "ocr": 0, "escaneado_omitido": 0, "corrupto_omitido": 0,
             "vacio_a_ocr": 0, "filtrado": 0, "ya_procesado": 0, "errores": 0}
    procesados = 0
    ocr_queue: list[str] = []   # PDFs que requieren OCR (escaneados + corruptos)

    for pdf_path in pdfs:
        meta = parse_meta_from_path(pdf_path, pdf_dir)

        # Filtros de alcance
        if types_filter and meta.doc_type not in types_filter:
            stats["filtrado"] += 1
            continue
        if years_filter and meta.year not in years_filter:
            stats["filtrado"] += 1
            continue
        if args.limit is not None and procesados >= args.limit:
            break

        # Salida espejando la estructura: data/extracted/<año>/<TIPO>/<archivo>.md
        rel_parent = pdf_path.parent.relative_to(pdf_dir)
        # Aplanar el "si/normatividad" intermedio para que la ruta sea legible
        clean_parts = [p for p in rel_parent.parts if p not in ("si", "normatividad")]
        out_subdir = output_dir.joinpath(*clean_parts)
        output_file = out_subdir / (pdf_path.stem + ".md")

        if output_file.exists():
            stats["ya_procesado"] += 1
            continue

        rel_display = pdf_path.relative_to(pdf_dir).as_posix()
        print(f"  [{rel_display}]")

        try:
            # Detectar el tipo de PDF para decidir qué extractor usar.
            digital = is_likely_digital(pdf_path)

            # --- Escaneado: necesita OCR (docling) ---
            if not digital:
                # Sin --ocr, los escaneados solo se anotan en la cola para la workstation.
                if not args.ocr:
                    print("    escaneado -> a cola de OCR (usar --ocr para procesarlo)")
                    stats["escaneado_omitido"] += 1
                    ocr_queue.append(rel_display)
                    continue
                print("    escaneado -> OCR (docling)...")
                doc = extract_scanned(pdf_path)
                method, kind = doc.method, "ocr"

            # --- Digital: texto seleccionable, se extrae con pymupdf4llm ---
            else:
                doc = extract_digital(pdf_path)
                method, kind = doc.method, "digital"

                # Fuente rota (letras cambiadas por dígitos): ningún extractor de texto
                # lo arregla -> se manda a OCR con docling.
                if doc.corrupt:
                    if args.ocr:
                        print("    fuente corrupta -> OCR...")
                        doc = extract_scanned(pdf_path)
                        method, kind = doc.method, "ocr"
                    else:
                        print("    fuente corrupta -> a cola de OCR (no se genera .md ruidoso)")
                        stats["corrupto_omitido"] += 1
                        ocr_queue.append(rel_display)
                        continue
                # Era 'digital' pero salió vacío: intentar OCR si está habilitado.
                elif doc.is_empty() and args.ocr:
                    print("    texto vacío -> intentando OCR...")
                    doc = extract_scanned(pdf_path)
                    method, kind = doc.method, "ocr"

            if doc.is_empty():
                # Clasificado como digital pero sin texto real extraíble (capa de texto
                # vacía). No es un error: se desvía a OCR como los escaneados.
                print("    sin texto extraíble -> a cola de OCR")
                stats["vacio_a_ocr"] += 1
                ocr_queue.append(rel_display)
                continue

            # Armar el .md (frontmatter + estructura legal) y guardarlo.
            md = to_markdown(doc.text, meta, method=method, pages=doc.pages)
            out_subdir.mkdir(parents=True, exist_ok=True)
            output_file.write_text(md, encoding="utf-8")

            print(f"    {kind} | {doc.pages} pág | {doc.char_count:,} chars -> {output_file.relative_to(output_dir).as_posix()}")
            stats[kind] += 1
            procesados += 1

        except Exception as e:
            print(f"    ERROR: {e}")
            stats["errores"] += 1

    # Cola de OCR para la workstation (escaneados + corruptos sin procesar).
    if ocr_queue and not args.ocr:
        queue_file = output_dir.parent / "_ocr_queue.txt"
        queue_file.write_text("\n".join(ocr_queue) + "\n", encoding="utf-8")
        print(f"\nCola de OCR ({len(ocr_queue)} PDFs) escrita en: {queue_file}")
        print("  -> correr en la workstation (GPU) con docling: python scripts/extract_text.py --ocr")

    print("\nResumen:")
    print(f"  Digitales procesados   : {stats['digital']}")
    print(f"  OCR procesados         : {stats['ocr']}")
    print(f"  Escaneados a cola OCR  : {stats['escaneado_omitido']}")
    print(f"  Corruptos a cola OCR   : {stats['corrupto_omitido']}")
    print(f"  Vacíos a cola OCR      : {stats['vacio_a_ocr']}")
    print(f"  Filtrados (fuera scope): {stats['filtrado']}")
    print(f"  Ya procesados (omit.)  : {stats['ya_procesado']}")
    print(f"  Errores                : {stats['errores']}")
    print(f"\nMarkdown guardado en: {output_dir}")


if __name__ == "__main__":
    main()
