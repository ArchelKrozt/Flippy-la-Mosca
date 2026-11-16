@echo off
rem FLIPPY LA MOSCA - doble clic para abrir (Windows)
rem La primera vez prepara un entorno de Python dentro de esta carpeta, instala lo necesario
rem (y el motor de GPU si hay una tarjeta NVIDIA) y descarga el conectoma, unos 135 MB.
rem Las siguientes veces abre directamente. Acepta las opciones de server.py, p. ej. --port 8080
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set "VPY=.venv\Scripts\python.exe"

if exist "%VPY%" goto gpu

rem --- buscar Python 3.10+ (el "python" de Microsoft Store no cuenta: falla al ejecutarlo)
set "PY="
py -3 -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul && set "PY=python"
if not defined PY (
    echo Necesitas Python 3.10 o superior: https://www.python.org/downloads/
    echo Al instalarlo marca la casilla "Add python.exe to PATH".
    goto fail
)
echo Primera vez: preparando Python para FLIPPY, 1-2 minutos...
%PY% -m venv .venv
if errorlevel 1 (
    echo No se pudo crear el entorno de Python.
    goto fail
)
"%VPY%" -m pip install -q --upgrade pip
"%VPY%" -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo No se pudieron instalar las dependencias.
    goto fail
)

:gpu
rem --- motor de GPU: solo con tarjeta NVIDIA, y una sola vez (si falla, se usa la CPU)
if exist ".venv\gpu-no" goto data
where nvidia-smi >nul 2>nul || goto data
"%VPY%" -c "import cupy" >nul 2>nul && goto data
echo Tarjeta NVIDIA detectada: instalando el motor de GPU, unos 2-3 minutos...
"%VPY%" -m pip install -q -r requirements-gpu.txt
if errorlevel 1 (
    echo No se pudo instalar el motor de GPU: FLIPPY usara la CPU.
    echo Para reintentarlo borra el archivo .venv\gpu-no
    type nul > ".venv\gpu-no"
)

:data
"%VPY%" download_data.py
if errorlevel 1 goto fail
"%VPY%" server.py %*
pause
exit /b 0

:fail
pause
exit /b 1
