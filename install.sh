#!/usr/bin/env bash
# =============================================================================
#  CANUTO — instalación del venv PRINCIPAL de extracción (Linux)
# =============================================================================
#  Prepara el pipeline PDF -> Markdown para PDFs DIGITALES y la generación del
#  dataset. NO instala torch (la extracción digital no lo usa).
#
#  El OCR de los escaneados (~73% del corpus) va en un venv APARTE con GPU:
#      bash install-docling.sh
#
#  Uso (desde la carpeta CANUTO/):
#      bash install.sh
# =============================================================================
set -e

echo "============================================================"
echo " CANUTO — venv principal (extracción digital + dataset)"
echo "============================================================"

PY="${PYTHON:-python3}"
echo "[Python] $($PY --version)"

echo "[1/3] Creando entorno virtual venv/ ..."
"$PY" -m venv venv

echo "[2/3] Actualizando pip ..."
./venv/bin/python -m pip install --upgrade pip

echo "[3/3] Instalando dependencias de extracción (sin torch) ..."
./venv/bin/pip install -r requirements-extract.txt

echo
echo "============================================================"
echo " Instalación completada"
echo "============================================================"
echo
echo "Siguientes pasos:"
echo "  source venv/bin/activate"
echo "  # copiar los PDFs a PDF/  (estructura <año>/si/normatividad/<TIPO>/)"
echo "  python scripts/extract_text.py                 # digitales -> data/extracted/*.md (+ cola OCR)"
echo "  python scripts/build_dataset.py --mode heuristic"
echo
echo "OCR de escaneados (GPU, venv aparte):"
echo "  bash install-docling.sh"
