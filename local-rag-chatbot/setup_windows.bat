@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher not found. Install Python 3.11 x64 first.
  pause
  exit /b 1
)
py -3.11 -m venv .venv
if errorlevel 1 (
  echo Could not create virtual environment. Confirm Python 3.11 is installed.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Package installation failed. Review the output above.
  pause
  exit /b 1
)
if not exist ".env" copy ".env.example" ".env"
echo Setup complete. Install Ollama and pull the models described in README.md.
pause
