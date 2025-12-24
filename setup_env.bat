@echo off
echo ============================================
echo VietTTS Desktop - Environment Setup
echo ============================================
echo.

:: Check Python version
python --version 2>NUL
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10 or higher from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Create virtual environment
echo [1/5] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

:: Activate virtual environment
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

:: Install PyTorch CPU version first
echo [3/5] Installing PyTorch (CPU version)...
pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

:: Install viettts from GitHub
echo [4/5] Installing VietTTS core engine...
pip install git+https://github.com/dangvansam/viet-tts.git

:: Install other requirements
echo [5/5] Installing other dependencies...
pip install -r requirements.txt

echo.
echo ============================================
echo Setup complete!
echo ============================================
echo.
echo To run the application:
echo   1. Activate: venv\Scripts\activate
echo   2. Run: python -m app.main
echo.
pause
