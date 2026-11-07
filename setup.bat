@echo off
REM One-time setup for Windows (CMD or PowerShell): makes .venv and installs packages.
echo Step 1: creating virtual environment in .venv ...
python -m venv .venv
if errorlevel 1 (
    echo Could not create .venv. Is Python installed and on PATH?
    exit /b 1
)
echo Step 2: installing packages from requirements.txt ...
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
echo.
echo Done! Activate with:  .venv\Scripts\activate
