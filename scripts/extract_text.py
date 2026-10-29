"""PDF -> Markdown (.md) estructurado, espejando la ruta del corpus SIRIUS.

Trabaja en dos etapas:
  1. PDF -> texto crudo: data/raw/<año>/<TIPO>/<n>_<año>.json
     Es lo lento (el OCR). Se hace UNA sola vez por PDF: si el crudo ya existe, no se
     vuelve a leer el PDF. Para repetir el OCR de un documento, borrar su .json.
  2. texto crudo -> .md: data/extracted/<año>/<TIPO>/<n>_<año>.md
     Con frontmatter YAML y la estructura legal en encabezados (CONSIDERANDO/RESUELVE ##,
     ARTICULO ###, PARAGRAFO ####). Tarda segundos y se rehace SIEMPRE, asi que una
     regla nueva de limpieza en to_markdown.py se aplica a todo el corpus sin repetir el OCR.

    python scripts/extract_text.py                        # solo digitales (escaneados -> cola OCR)
    python scripts/extract_text.py --ocr                  # + OCR docling (workstation con GPU)
    python scripts/extract_text.py --solo-md              # rehace los .md desde el crudo, sin abrir PDFs
    python scripts/extract_text.py --lista data/_ocr_queue.txt --ocr   # solo los PDFs de la lista
    python scripts/extract_text.py --types RESOLUCION_SUPERIOR --years 2024,2025 --limit 10
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.extraction.pdf_digital import extract as extract_digital, is_likely_digital
from src.extraction.pdf_scanned import extract as extract_scanned
from src.extraction.to_markdown import parse_meta_from_path, to_markdown


def _parse_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {v.strip() for v in value.split(",") if v.strip()}


def _leer_lista(path: str | None) -> set[str] | None:
    # Una ruta relativa a PDF/ por linea, como en data/_ocr_queue.txt
    if not path:
        return None
    lineas = Path(path).read_text(encoding="utf-8").splitlines()
    return {l.strip().replace("\\", "/") for l in lineas if l.strip()}


def _guardar_crudo(raw_file: Path, doc, method: str):
    # El crudo es la salida tal cual del extractor, sin ninguna limpieza.
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    datos = {"metodo": method, "paginas": doc.pages, "texto": doc.text}
    raw_file.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")


def _extraer_pdf(pdf_path: Path, usar_ocr: bool):
    """Etapa 1: lee el PDF. Devuelve (doc, metodo, motivo).

    Si el PDF necesita OCR y no se pidio --ocr, doc es None y motivo dice por que
    (escaneado, corrupto o vacio) para mandarlo a la cola.
    """
    if not is_likely_digital(pdf_path):
        # Sin --ocr, los escaneados solo se anotan en la cola para la workstation.
        if not usar_ocr:
            print("    escaneado -> a cola de OCR (usar --ocr para procesarlo)")
            return None, None, "escaneado_omitido"
        print("    escaneado -> OCR (docling)...")
        doc = extract_scanned(pdf_path)
        return doc, doc.method, None

    # Digital: texto seleccionable, se extrae con pymupdf4llm.
    doc = extract_digital(pdf_path)

    # Fuente rota (letras cambiadas por digitos): ningun extractor de texto
    # lo arregla -> se manda a OCR con docling.
    if doc.corrupt:
        if not usar_ocr:
            print("    fuente corrupta -> a cola de OCR (no se genera .md ruidoso)")
            return None, None, "corrupto_omitido"
        print("    fuente corrupta -> OCR...")
        doc = extract_scanned(pdf_path)
    # Era 'digital' pero salio vacio: intentar OCR si esta habilitado.
    elif doc.is_empty() and usar_ocr:
        print("    texto vacío -> intentando OCR...")
        doc = extract_scanned(pdf_path)

    if doc.is_empty():
        # Clasificado como digital pero sin texto real extraible (capa de texto
        # vacia). No es un error: se desvia a OCR como los escaneados.
        print("    sin texto extraíble -> a cola de OCR")
        return None, None, "vacio_a_ocr"

    return doc, doc.method, None


def main():
    parser = argparse.ArgumentParser(description="Extrae texto de PDFs normativos a Markdown.")
    parser.add_argument("--pdf-dir", default=None, help="Directorio raíz del corpus de PDFs")
    parser.add_argument("--raw-dir", default=None, help="Directorio del texto crudo (.json)")
    parser.add_argument("--output-dir", default=None, help="Directorio de salida para .md")
    parser.add_argument("--ocr", action="store_true",
                        help="Procesar también PDFs escaneados/corruptos con OCR docling (usar en la workstation con GPU)")
    parser.add_argument("--solo-md", action="store_true",
                        help="Solo etapa 2: rehacer los .md desde el texto crudo, sin abrir ningún PDF")
    parser.add_argument("--lista", default=None,
                        help="Archivo con las rutas de los PDFs a procesar, una por línea (ej. data/_ocr_queue.txt)")
    parser.add_argument("--types", default=None,
                        help="Lista separada por comas de tipos a incluir (ej. RESOLUCION_ACADEMICA)")
    parser.add_argument("--years", default=None,
                        help="Lista separada por comas de años a incluir (ej. 2024,2025,2026)")
    parser.add_argument("--limit", type=int, default=None, help="Procesar a lo sumo N PDFs (pruebas)")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ext_cfg = config.get("extraction", {})

    root = Path(__file__).parent.parent
    pdf_dir = Path(args.pdf_dir or ext_cfg.get("pdf_dir", "PDF"))
    raw_dir = Path(args.raw_dir or ext_cfg.get("raw_dir", "data/raw"))
    output_dir = Path(args.output_dir or ext_cfg.get("output_dir", "data/extracted"))
    if not pdf_dir.is_absolute():
        pdf_dir = (root / pdf_dir).resolve()
    if not raw_dir.is_absolute():
        raw_dir = (root / raw_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()

    types_filter = _parse_csv(args.types)
    years_filter = _parse_csv(args.years)
    lista = _leer_lista(args.lista)

    pdfs = sorted(p for p in pdf_dir.rglob("*.pdf"))
    if not pdfs:
        print(f"No se encontraron PDFs en: {pdf_dir}")
        return

    print(f"Corpus: {pdf_dir}")
    print(f"Crudo : {raw_dir}")
    print(f"Salida: {output_dir}")
    if args.solo_md:
        print("Modo: solo .md desde el crudo (no se abre ningún PDF)")
    else:
        print(f"OCR escaneados: {'SÍ (docling)' if args.ocr else 'NO (se omiten)'}")
    if types_filter:
        print(f"Filtro tipos: {', '.join(sorted(types_filter))}")
    if years_filter:
        print(f"Filtro años: {', '.join(sorted(years_filter))}")
    if lista is not None:
        print(f"Filtro lista: {len(lista)} PDFs de {args.lista}")
    print()

    stats = {"digital": 0, "ocr": 0, "desde_crudo": 0, "sin_crudo": 0,
             "escaneado_omitido": 0, "corrupto_omitido": 0, "vacio_a_ocr": 0,
             "filtrado": 0, "errores": 0}
    procesados = 0
    ocr_queue: list[str] = []   # PDFs que requieren OCR (escaneados + corruptos)

    for pdf_path in pdfs:
        meta = parse_meta_from_path(pdf_path, pdf_dir)
        rel_display = pdf_path.relative_to(pdf_dir).as_posix()

        if types_filter and meta.doc_type not in types_filter:
            stats["filtrado"] += 1
            continue
        if years_filter and meta.year not in years_filter:
            stats["filtrado"] += 1
            continue
        if lista is not None and rel_display not in lista:
            stats["filtrado"] += 1
            continue
        if args.limit is not None and procesados >= args.limit:
            break

        # Rutas espejando la estructura: <año>/<TIPO>/<archivo>
        rel_parent = pdf_path.parent.relative_to(pdf_dir)
        # Aplanar el "si/normatividad" intermedio para que la ruta sea legible
        clean_parts = [p for p in rel_parent.parts if p not in ("si", "normatividad")]
        raw_file = raw_dir.joinpath(*clean_parts) / (pdf_path.stem + ".json")
        out_subdir = output_dir.joinpath(*clean_parts)
        output_file = out_subdir / (pdf_path.stem + ".md")

        try:
            # El crudo se extrae una sola vez; el .md se rehace siempre.
            if raw_file.exists():
                datos = json.loads(raw_file.read_text(encoding="utf-8"))
                text, method, pages = datos["texto"], datos["metodo"], datos["paginas"]
                kind = "desde_crudo"
            elif args.solo_md:
                stats["sin_crudo"] += 1
                continue
            else:
                print(f"  [{rel_display}]")
                doc, method, motivo = _extraer_pdf(pdf_path, args.ocr)
                if doc is None:
                    stats[motivo] += 1
                    ocr_queue.append(rel_display)
                    continue
                _guardar_crudo(raw_file, doc, method)
                text, pages = doc.text, doc.pages
                kind = "digital" if method != "docling" else "ocr"

            md = to_markdown(text, meta, method=method, pages=pages)
            out_subdir.mkdir(parents=True, exist_ok=True)
            output_file.write_text(md, encoding="utf-8")

            if kind != "desde_crudo":
                print(f"    {kind} | {pages} pág | {len(text):,} chars -> {output_file.relative_to(output_dir).as_posix()}")
            stats[kind] += 1
            procesados += 1

        except Exception as e:
            print(f"  [{rel_display}]")
            print(f"    ERROR: {e}")
            stats["errores"] += 1

    # Cola de OCR para la workstation (escaneados + corruptos sin procesar).
    if ocr_queue and not args.ocr:
        queue_file = output_dir.parent / "_ocr_queue.txt"
        queue_file.write_text("\n".join(ocr_queue) + "\n", encoding="utf-8")
        print(f"\nCola de OCR ({len(ocr_queue)} PDFs) escrita en: {queue_file}")
        print("  -> correr en la workstation (GPU) con docling: python scripts/extract_text.py --ocr")

    print("\nResumen:")
    print(f"  Digitales extraídos    : {stats['digital']}")
    print(f"  OCR extraídos          : {stats['ocr']}")
    print(f"  .md rehechos del crudo : {stats['desde_crudo']}")
    print(f"  Sin crudo (--solo-md)  : {stats['sin_crudo']}")
    print(f"  Escaneados a cola OCR  : {stats['escaneado_omitido']}")
    print(f"  Corruptos a cola OCR   : {stats['corrupto_omitido']}")
    print(f"  Vacíos a cola OCR      : {stats['vacio_a_ocr']}")
    print(f"  Filtrados (fuera scope): {stats['filtrado']}")
    print(f"  Errores                : {stats['errores']}")
    print(f"\nTexto crudo en : {raw_dir}")
    print(f"Markdown en    : {output_dir}")


if __name__ == "__main__":
    main()
