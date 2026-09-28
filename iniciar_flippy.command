#!/bin/bash
# FLIPPY LA MOSCA · doble clic para abrir (macOS; en Linux: ./iniciar_flippy.command)
# La primera vez prepara un entorno de Python dentro de esta carpeta, instala lo necesario
# y descarga el conectoma (~135 MB). Las siguientes veces abre directamente.
cd "$(dirname "$0")" || exit 1
pausa() { read -r -p "Pulsa Enter para cerrar esta ventana..." _; }

if ! command -v python3 >/dev/null 2>&1; then
  echo "Necesitas Python 3.10 o superior: https://www.python.org/downloads/"
  pausa; exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "Primera vez: preparando Python para FLIPPY (1-2 minutos)..."
  python3 -m venv .venv || { echo "No se pudo crear el entorno de Python."; pausa; exit 1; }
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r requirements.txt || { echo "No se pudieron instalar las dependencias."; pausa; exit 1; }
fi
.venv/bin/python download_data.py || { pausa; exit 1; }
.venv/bin/python server.py
pausa
