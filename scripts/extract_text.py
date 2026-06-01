"""Extrae texto de todos los PDFs en un directorio y los guarda como archivos .txt.

Uso:
    python scripts/extract_text.py
    python scripts/extract_text.py --pdf-dir ../PDF --output-dir data/extracted
    python scripts/extract_text.py --ocr-engine tesseract

El script detecta automáticamente si cada PDF es digital o escaneado.
Los archivos ya procesados se omiten (para reejecutar, borra el .txt correspondiente).
"""
import sys
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Forzar UTF-8 en stdout para evitar errores con tildes y caracteres especiales en Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
from src.config_loader import load_config
from src.extraction.pdf_digital import extract as extract_digital, is_likely_digital
from src.extraction.pdf_scanned import extract as extract_scanned


def main():
    parser = argparse.ArgumentParser(description="Extrae texto de PDFs normativos.")
    parser.add_argument("--pdf-dir", default=None, help="Directorio con PDFs fuente")
    parser.add_argument("--output-dir", default=None, help="Directorio de salida para .txt")
    parser.add_argument("--ocr-engine", default=None, choices=["easyocr", "tesseract"])
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ext_cfg = config.get("extraction", {})

    pdf_dir = Path(args.pdf_dir or ext_cfg.get("pdf_dir", "../PDF"))
    output_dir = Path(args.output_dir or ext_cfg.get("output_dir", "data/extracted"))
    ocr_engine = args.ocr_engine or ext_cfg.get("ocr_engine", "easyocr")

    root = Path(__file__).parent.parent
    if not pdf_dir.is_absolute():
        pdf_dir = (root / pdf_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Usar set para evitar duplicados en Windows (filesystem case-insensitive)
    pdfs = sorted({p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf"})
    if not pdfs:
        print(f"No se encontraron PDFs en: {pdf_dir}")
        return

    print(f"Procesando {len(pdfs)} PDF(s) desde: {pdf_dir}")
    print(f"Salida en: {output_dir}\n")

    stats = {"digital": 0, "escaneado": 0, "errores": 0}

    for pdf_path in pdfs:
        print(f"  [{pdf_path.name}]")
        output_file = output_dir / (pdf_path.stem + ".txt")

        if output_file.exists():
            print("    ya procesado, omitiendo.")
            continue

        try:
            digital = is_likely_digital(pdf_path)
            if digital:
                doc = extract_digital(pdf_path)
                kind = "digital"
            else:
                print(f"    escaneado detectado - usando OCR ({ocr_engine})...")
                doc = extract_scanned(pdf_path, engine=ocr_engine)
                kind = "escaneado"

            # Si la extracción digital dio resultado vacío, intentar OCR
            if doc.is_empty() and kind == "digital":
                print(f"    texto vacio con extraccion digital, intentando OCR...")
                doc = extract_scanned(pdf_path, engine=ocr_engine)
                kind = "escaneado"

            output_file.write_text(doc.text, encoding="utf-8")
            print(f"    {kind} | {doc.pages} pag | {doc.char_count:,} chars -> {output_file.name}")
            stats[kind] += 1

        except Exception as e:
            print(f"    ERROR: {e}")
            stats["errores"] += 1

    print(f"\nResumen:")
    print(f"  Digitales procesados : {stats['digital']}")
    print(f"  Escaneados (OCR)     : {stats['escaneado']}")
    print(f"  Errores              : {stats['errores']}")
    print(f"\nTextos guardados en: {output_dir}")


if __name__ == "__main__":
    main()
