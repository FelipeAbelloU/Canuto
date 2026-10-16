"""Lista los documentos cuyo numero impreso no coincide con el nombre del archivo.

El numero de la cita sale del nombre del PDF (49_2015.pdf -> Resolucion Superior 49 de 2015),
pero algunos archivos estan mal nombrados: adentro dice "RESOLUCION SUPERIOR No. 050 DE 2015".
extract_text.py guarda el numero impreso en el frontmatter de cada .md (numero_impreso,
anio_impreso); este script los compara y escribe la lista para revisarla a mano contra el PDF.

    python scripts/revisar_numeros.py
    python scripts/revisar_numeros.py --md-dir data/extracted_nuevo
    -> data/numeros_por_revisar.tsv
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_dataset import _split_frontmatter
from src.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description="Compara el numero impreso con el del archivo.")
    parser.add_argument("--md-dir", default=None, help="Carpeta de los .md (por defecto la de config.yaml)")
    args = parser.parse_args()

    config = load_config("config/config.yaml")
    root = Path(__file__).parent.parent
    md_dir = Path(args.md_dir or config.get("extraction", {}).get("output_dir", "data/extracted"))
    if not md_dir.is_absolute():
        md_dir = root / md_dir

    distintos = []
    total = con_impreso = 0
    for f in sorted(md_dir.rglob("*.md")):
        meta, _ = _split_frontmatter(f.read_text(encoding="utf-8"))
        total += 1
        num_imp, anio_imp = meta.get("numero_impreso", ""), meta.get("anio_impreso", "")
        if not num_imp:
            continue
        con_impreso += 1
        num_archivo = meta.get("numero", "").lstrip("0") or meta.get("numero", "")
        if num_imp != num_archivo or anio_imp != meta.get("anio", ""):
            distintos.append([meta.get("ruta_origen", f.name), meta.get("tipo", ""),
                              meta.get("numero", ""), meta.get("anio", ""), num_imp, anio_imp])

    salida = root / "data" / "numeros_por_revisar.tsv"
    filas = ["archivo\ttipo\tnumero_archivo\tanio_archivo\tnumero_impreso\tanio_impreso"]
    filas += ["\t".join(d) for d in distintos]
    salida.write_text("\n".join(filas) + "\n", encoding="utf-8")

    print(f"Documentos: {total} | con numero impreso: {con_impreso} | distintos: {len(distintos)}")
    for d in distintos:
        print(f"  {d[0]}: archivo {d[2]} de {d[3]} -> impreso {d[4]} de {d[5]}")
    print(f"\nLista en: {salida}")


if __name__ == "__main__":
    main()
