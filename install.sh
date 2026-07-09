#!/bin/bash
# Instala el entorno principal de CANUTO en la workstation (Ubuntu, GPU RTX 5080).
# Sirve para todo: extraer PDFs, armar el dataset, entrenar, inferir y la web.
# El OCR con docling va en un entorno aparte: usar install-docling.sh.
#
# Clonar el proyecto dentro de /bodega (hay espacio) y correr desde ahi:
#   bash install.sh

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# PyTorch para la RTX 5080 (Blackwell -> CUDA 13, wheels cu130)
pip install torch --index-url https://download.pytorch.org/whl/cu130

# Resto de dependencias del proyecto
pip install -r requirements.txt

# Comprobacion rapida de que la GPU se ve
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"

echo "Listo. Activar con: source venv/bin/activate"
