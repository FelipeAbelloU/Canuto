@echo off
REM =============================================================
REM  Asistente Normativo Unillanos -- Instalacion del entorno
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

echo [1/4] Creando entorno virtual...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] No se pudo crear el entorno virtual.
    pause
    exit /b 1
)

echo [2/4] Activando entorno virtual...
call venv\Scripts\activate.bat

echo [3/4] Instalando PyTorch...
echo.
echo  Elige el tipo de instalacion:
echo    1. Solo CPU (laptop i5-7200U)
echo    2. CUDA 12.1 (workstation RTX 4090i)
echo.
set /p opcion="Opcion (1 o 2): "

if "%opcion%"=="2" (
    echo Instalando PyTorch con soporte CUDA 12.1...
    pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
) else (
    echo Instalando PyTorch solo CPU...
    pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
)

if errorlevel 1 (
    echo [ERROR] Fallo al instalar PyTorch.
    pause
    exit /b 1
)

echo [4/4] Instalando dependencias del proyecto...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Fallo al instalar dependencias.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Instalacion completada
echo ============================================================
echo.
echo Pasos siguientes:
echo.
echo  1. Extraer texto de los PDFs normativos:
echo     venv\Scripts\python scripts\extract_text.py
echo.
echo  2. Generar dataset de fine-tuning:
echo     venv\Scripts\python scripts\build_dataset.py --mode heuristic
echo.
echo  3. Probar el chat por terminal (requiere checkpoint entrenado):
echo     venv\Scripts\python scripts\chat_cli.py
echo.
echo  4. Abrir la interfaz web:
echo     venv\Scripts\python ui\app.py
echo.
echo  5. Iniciar la API REST para SIRIUS:
echo     venv\Scripts\uvicorn src.api.app:app --reload --port 8000
echo.
echo  NOTA: El modelo fine-tuneado debe configurarse en config\config.yaml
echo        bajo model.checkpoint_path antes de usar el chat.
echo.
pause
