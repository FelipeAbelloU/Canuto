#!/usr/bin/env bash
# =============================================================================
#  CANUTO — instalación del venv de OCR con docling (Linux + GPU)
# =============================================================================
#  Procesa los PDFs ESCANEADOS (~73% del corpus) con OCR docling en GPU.
#  Entorno APARTE del principal: docling exige transformers moderno que rompe
#  el stack de inferencia/entrenamiento.
#
#  Workstation objetivo: Linux, Python 3.12.9, driver 595 / CUDA 13.2.
#  Wheel de torch = cu126 (única línea que publica torch 2.12 para cp312; el
#  driver CUDA 13.2 corre cu126 sin problema, es la dirección forward-compatible).
#
#  Uso (desde la carpeta CANUTO/):
#      bash install-docling.sh
# =============================================================================
set -e

echo "============================================================"
echo " CANUTO — venv-docling (OCR de escaneados, GPU)"
echo "============================================================"

PY="${PYTHON:-python3}"
echo "[Python] $($PY --version)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi | head -n 4 || echo "[aviso] nvidia-smi no encontrado; ¿hay GPU/driver?"

echo "[1/4] Creando entorno virtual venv-docling/ ..."
"$PY" -m venv venv-docling

echo "[2/4] Actualizando pip ..."
./venv-docling/bin/python -m pip install --upgrade pip

echo "[3/4] Instalando PyTorch CUDA (cu126) ..."
./venv-docling/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126

echo "[4/4] Instalando docling y utilidades (torch se resuelve desde cu126) ..."
./venv-docling/bin/pip install -r requirements-docling.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126

echo
echo "Verificando que torch ve la GPU ..."
if ./venv-docling/bin/python -c "import torch,sys; print('torch', torch.__version__, '| CUDA disponible:', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "OK: la GPU es visible para torch."
else
    echo "-------------------------------------------------------------------"
    echo "AVISO: torch NO ve la GPU (probablemente se instaló una build CPU"
    echo "porque docling movió la versión de torch). Reinstala desde cu126:"
    echo "  ./venv-docling/bin/pip install --force-reinstall torch \\"
    echo "      --index-url https://download.pytorch.org/whl/cu126"
    echo "y vuelve a verificar."
    echo "-------------------------------------------------------------------"
fi

echo
echo "Listo. Para OCR-izar los escaneados (MISMO output-dir que el venv principal):"
echo "  source venv-docling/bin/activate"
echo "  python scripts/extract_text.py --ocr"
