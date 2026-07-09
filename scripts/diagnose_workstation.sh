#!/usr/bin/env bash
# =============================================================================
#  CANUTO - Diagnostico de la workstation (Ubuntu)  [SOLO LECTURA]
# =============================================================================
#  No instala, no modifica, no requiere sudo. Solo recoge informacion del
#  entorno para decidir versiones/compatibilidad antes de montar el proyecto.
#
#  Uso:
#     bash diagnose_workstation.sh              # imprime en pantalla
#     bash diagnose_workstation.sh | tee canuto_diag.txt   # + guarda a archivo
#  Luego envia el archivo canuto_diag.txt (o copia lo que salga en pantalla).
# =============================================================================

# have <cmd> -> true si el comando existe (para no romper si algo falta)
have() { command -v "$1" >/dev/null 2>&1; }
# seccion <titulo> -> imprime un encabezado legible
seccion() { echo; echo "==================== $1 ===================="; }
# muestra <cmd...> -> corre el comando o avisa si no esta disponible
muestra() { if have "$1"; then "$@"; else echo "  [no disponible: $1]"; fi; }

echo "############################################################"
echo "#  CANUTO - Diagnostico de workstation"
echo "#  Fecha: $(date)"
echo "#  Host:  $(hostname 2>/dev/null)   Usuario: $(whoami 2>/dev/null)"
echo "############################################################"

seccion "SISTEMA OPERATIVO"
if [ -f /etc/os-release ]; then cat /etc/os-release; fi
muestra lsb_release -a 2>/dev/null
echo "-- Kernel / arquitectura --"
uname -a

seccion "glibc (define que 'wheels' de Python sirven)"
# manylinux_2_28 (lo que usan PyMuPDF/torch modernos) exige glibc >= 2.28
muestra ldd --version | head -n 1

seccion "CPU"
if have lscpu; then lscpu | grep -Ei 'model name|^cpu\(s\)|core|thread|arch'; else cat /proc/cpuinfo | grep -m1 'model name'; fi
echo "-- Nucleos logicos (nproc) --"
muestra nproc

seccion "MEMORIA RAM"
muestra free -h
echo "-- MemTotal exacto --"
grep -E 'MemTotal|MemAvailable' /proc/meminfo 2>/dev/null

seccion "DISCO (espacio libre - importante para modelos y cache HF)"
# El entrenamiento + cache de HuggingFace + checkpoints pueden pesar decenas de GB
df -h / /home /tmp "$HOME" "$PWD" 2>/dev/null | sort -u

seccion "GPU (NVIDIA)"
if have nvidia-smi; then
    nvidia-smi
    echo "-- Resumen GPU --"
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
else
    echo "  [nvidia-smi NO encontrado -> revisar driver NVIDIA]"
fi

seccion "CUDA toolkit (nvcc) - puede diferir del CUDA del driver"
# bitsandbytes/torch pueden ser sensibles a esto
muestra nvcc --version
echo "-- Librerias CUDA runtime visibles --"
if have ldconfig; then ldconfig -p | grep -Ei 'libcudart|libcublas|libcudnn' | head -n 10 || echo "  [ninguna en el sistema; torch trae las suyas en el wheel - OK]"; fi

seccion "PYTHON"
echo "-- python3 --"; muestra python3 --version; echo "   ruta: $(command -v python3 2>/dev/null)"
echo "-- python (por si acaso) --"; muestra python --version
echo "-- pip3 --"; muestra pip3 --version
echo "-- modulo venv (necesario para crear entornos) --"
if have python3; then python3 -c "import venv; print('  venv: OK')" 2>/dev/null || echo "  venv: FALTA (paquete python3-venv)"; fi
echo "-- cabeceras de desarrollo (python3-dev) --"
PYINC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null)
if [ -n "$PYINC" ] && [ -f "$PYINC/Python.h" ]; then echo "  python3-dev: OK ($PYINC/Python.h)"; else echo "  python3-dev: no detectado (quizas no haga falta)"; fi

seccion "HERRAMIENTAS DE COMPILACION (por si algun paquete compila)"
muestra gcc --version | head -n 1
muestra g++ --version | head -n 1
muestra make --version | head -n 1
muestra cmake --version | head -n 1

seccion "GIT"
muestra git --version

seccion "ENTORNOS EXISTENTES (conda / otros)"
muestra conda --version
echo "-- carpetas de venv/conda que ya existan en HOME --"
ls -d "$HOME"/miniconda3 "$HOME"/anaconda3 "$HOME"/venv* "$HOME"/.conda 2>/dev/null || echo "  (ninguna)"

seccion "RED (puede llegar a los repos de paquetes?)"
# Sin acceso a estos, pip/HF fallarian. Tambien detecta si hay proxy.
echo "-- Variables de proxy --"
env | grep -Ei 'http_proxy|https_proxy|no_proxy' || echo "  (sin proxy configurado)"
for url in https://pypi.org https://files.pythonhosted.org https://download.pytorch.org https://huggingface.co; do
    if have curl; then
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$url" 2>/dev/null)
        echo "  $url -> HTTP $code"
    elif have wget; then
        wget -q --timeout=10 --spider "$url" && echo "  $url -> OK" || echo "  $url -> FALLO"
    else
        echo "  [ni curl ni wget disponibles para probar $url]"
    fi
done

seccion "LOCALE / UTF-8"
muestra locale | grep -E 'LANG|LC_ALL'

echo
echo "############################################################"
echo "#  Fin del diagnostico. Envia esta salida (o canuto_diag.txt)."
echo "############################################################"
