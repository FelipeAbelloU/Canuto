@echo off
REM =============================================================
REM  Asistente Normativo Unillanos — Instalacion del entorno
REM  Ejecutar UNA SOLA VEZ desde la carpeta Proyecto5\
REM =============================================================

echo.
echo ============================================================
echo  Asistente Normativo Unillanos - Instalacion
echo ============================================================
echo.

REM --- Verificar Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado. Instala Python 3.11+ desde python.org
    pause
    exit /b 1
)

echo [1/5] Creando entorno virtual...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] No se pudo crear el entorno virtual.
    pause
    exit /b 1
)

echo [2/5] Activando entorno virtual...
call venv\Scripts\activate.bat

echo [3/5] Instalando PyTorch (solo CPU, optimizado para laptop)...
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
if errorlevel 1 (
    echo [ERROR] Fallo al instalar PyTorch.
    pause
    exit /b 1
)

echo [4/5] Instalando dependencias del proyecto...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Fallo al instalar dependencias.
    pause
    exit /b 1
)

echo [5/5] Copiando archivo de variables de entorno...
if not exist .env (
    copy .env.example .env >nul
    echo    Archivo .env creado. Edita las claves API si las necesitas.
)

echo.
echo ============================================================
echo  Instalacion completada
echo ============================================================
echo.
echo Pasos siguientes:
echo.
echo  1. Indexar los documentos normativos (PDFs de Tesis\PDF\):
echo     venv\Scripts\python scripts\ingest.py --docs-dir ..\PDF
echo.
echo  2. Probar el chat en la terminal:
echo     venv\Scripts\python scripts\chat_cli.py
echo.
echo  3. Abrir la interfaz web (Gradio):
echo     venv\Scripts\python ui\app.py
echo.
echo  4. (Opcional) Para respuestas generadas, instala Ollama:
echo     https://ollama.com
echo     Luego: ollama pull qwen2.5:1.5b
echo     Y cambia en config\config.yaml:  backend: ollama
echo.
pause
