#!/usr/bin/env bash
# One-time setup for Git Bash, Linux, macOS or WSL: makes .venv and installs packages.
set -e

echo "Step 1: creating virtual environment in .venv ..."
python -m venv .venv || python3 -m venv .venv

# The activate script is in Scripts/ on Windows and in bin/ on Linux/macOS
if [ -f .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python; else PY=.venv/bin/python; fi

echo "Step 2: installing packages from requirements.txt ..."
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements.txt

echo
echo "Done! Activate with:"
echo "  Git Bash: source .venv/Scripts/activate"
echo "  Linux/macOS/WSL: source .venv/bin/activate"
