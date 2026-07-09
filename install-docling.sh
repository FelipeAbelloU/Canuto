#!/bin/bash
# Instala el entorno del OCR con docling (entorno aparte, usa la GPU).
# docling necesita versiones de librerias que chocan con el entorno principal,
# por eso va separado. Correr despues de install.sh.
#
#   bash install-docling.sh

python3 -m venv venv-docling
source venv-docling/bin/activate
pip install --upgrade pip

# PyTorch para la RTX 5080 (Blackwell -> CUDA 13, wheels cu130)
pip install torch --index-url https://download.pytorch.org/whl/cu130

# docling + utilidades (el --extra-index-url mantiene torch en la version de GPU)
pip install -r requirements-docling.txt --extra-index-url https://download.pytorch.org/whl/cu130

python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"

echo "Listo. OCR: source venv-docling/bin/activate && python scripts/extract_text.py --ocr"
