"""Baja de SIRIUS los PDF vigentes que faltan en PDF/, segun el Excel exportado de si_normatividad.

Guarda cada PDF en PDF/ con la misma ruta de archivo_soporte (2024/si/normatividad/<TIPO>/<n>_<año>.pdf),
asi extract_text.py lo reconoce sin cambios. Tambien vuelve a bajar los PDF locales que estan cortados.
Escribe data/_sirius_nuevos.txt para pasarlo a extract_text.py --lista.

    python scripts/descargar_sirius.py --excel data/sirius/Normatividad-2026-10-30.xlsx --solo-listar
    python scripts/descargar_sirius.py --excel data/sirius/Normatividad-2026-10-30.xlsx
"""
import sys
import time
import argparse
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config

URL_BASE = "https://sirius.unillanos.edu.co/media/"
ANIO_DESDE = 2015
ANIO_HASTA = 2026
ESTADO_VIGENTE = 5
# codigos de tpnormatividad: AS 4, AA 5, RS 16, RA 6, RR 7, FCBI 8, ley 1, decreto MEN 2,
# decreto nacional 15, res. MEN 3, CESU 14, res. min. ambiente 13
# fuera: plan 17, ordenanza 18, conpes 19 y actas 20 y 21
TIPOS_PERMITIDOS = {4, 5, 16, 6, 7, 8, 1, 2, 15, 3, 14, 13}
# sirius cambio la carpeta de algunas normas nacionales; las de PDF/ tienen el nombre viejo
CARPETA_VIEJA = {
    "DECRETO_MEN": "DECRETO_MINISTERIO_DE_EDUCACION_NACIONAL",
    "RESOLUCION_MEN": "RESOLUCION_MINISTERIO_DE_EDUCACION_NACIONAL",
}
COLUMNAS = ("year", "numero", "estado", "tpnormatividad", "archivo_soporte")
PAUSA = 1.0


def leer_excel(ruta):
    hoja = openpyxl.load_workbook(ruta, read_only=True).active
    filas = hoja.iter_rows(values_only=True)
    encabezado = list(next(filas))
    faltan = [c for c in COLUMNAS if c not in encabezado]
    if faltan:
        sys.exit(f"Al Excel le faltan las columnas: {', '.join(faltan)}")
    pos = {c: encabezado.index(c) for c in COLUMNAS}
    return [{c: fila[pos[c]] for c in COLUMNAS} for fila in filas]


def ruta_local(fila):
    rel = fila["archivo_soporte"]
    # en sirius hay rutas sin un año valido (217/.../8_217.pdf); se guardan con el del excel
    if len(rel.split("/")[0]) != 4:
        carpeta = rel.split("/")[3]
        rel = f"{fila['year']}/si/normatividad/{carpeta}/{fila['numero']}_{fila['year']}.pdf"
    return rel


def completo(ruta):
    # un pdf cortado no trae la marca %%EOF al final (asi llego la RA 63/2022)
    with open(ruta, "rb") as f:
        inicio = f.read(5)
        f.seek(max(0, ruta.stat().st_size - 2048))
        return inicio.startswith(b"%PDF") and b"%%EOF" in f.read()


def bajar(url, destino):
    parcial = destino.with_suffix(".part")
    pedido = urllib.request.Request(url, headers={"User-Agent": "CANUTO-descarga"})
    for intento in (1, 2):
        try:
            with urllib.request.urlopen(pedido, timeout=60) as r:
                parcial.write_bytes(r.read())
            break
        except urllib.error.HTTPError as e:
            # 404: el registro esta en el excel pero el pdf no esta en el servidor
            return f"no esta en sirius ({e.code})"
        except (urllib.error.URLError, TimeoutError) as e:
            if intento == 2:
                return f"error: {e}"
            time.sleep(5)
    # se renombra solo si llego entero; si se va la luz no queda un pdf a medias
    if not completo(parcial):
        parcial.unlink()
        return "llego incompleto"
    parcial.replace(destino)
    return "ok"


def main():
    parser = argparse.ArgumentParser(description="Baja de SIRIUS los PDF vigentes que faltan.")
    parser.add_argument("--excel", required=True, help="Excel exportado de si_normatividad")
    parser.add_argument("--solo-listar", action="store_true", help="Muestra lo que bajaria, sin descargar")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    config = load_config("config/config.yaml")
    pdf_dir = root / config.get("extraction", {}).get("pdf_dir", "PDF")

    # en linux las mayusculas importan; asi no se baja dos veces un pdf que ya esta con otro nombre
    locales = {p.relative_to(pdf_dir).as_posix().lower(): p for p in pdf_dir.rglob("*.pdf")}

    nuevos, cortados = [], []
    for fila in leer_excel(args.excel):
        if not (ANIO_DESDE <= fila["year"] <= ANIO_HASTA and fila["estado"] == ESTADO_VIGENTE
                and fila["tpnormatividad"] in TIPOS_PERMITIDOS):
            continue
        rel = ruta_local(fila)
        carpeta = rel.split("/")[3]
        rel_vieja = rel.replace(f"/{carpeta}/", f"/{CARPETA_VIEJA.get(carpeta, carpeta)}/")
        local = locales.get(rel.lower()) or locales.get(rel_vieja.lower())
        if local is None:
            nuevos.append((fila["archivo_soporte"], pdf_dir / rel))
        elif not completo(local):
            cortados.append((fila["archivo_soporte"], local))

    print(f"Por bajar: {len(nuevos)} nuevos y {len(cortados)} cortados")
    if args.solo_listar:
        for _, destino in nuevos:
            print(f"  nuevo    {destino.relative_to(pdf_dir).as_posix()}")
        for _, destino in cortados:
            print(f"  cortado  {destino.relative_to(pdf_dir).as_posix()}")
        return

    bajados, fallidos = [], []
    for rel, destino in nuevos + cortados:
        url = URL_BASE + urllib.parse.quote(rel)
        destino.parent.mkdir(parents=True, exist_ok=True)
        resultado = bajar(url, destino)
        print(f"  {resultado:20} {rel}")
        if resultado == "ok":
            bajados.append(destino.relative_to(pdf_dir).as_posix())
        else:
            fallidos.append((rel, url, resultado))
        time.sleep(PAUSA)

    lista = root / "data/_sirius_nuevos.txt"
    lista.write_text("".join(f"{r}\n" for r in bajados), encoding="utf-8")
    print(f"\nBajados: {len(bajados)}  fallidos: {len(fallidos)}  -> {lista.relative_to(root)}")
    for rel, url, motivo in fallidos:
        print(f"  FALLO ({motivo}): {url}")


if __name__ == "__main__":
    main()
