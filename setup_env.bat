@echo off
echo ============================================
echo Vietnamese TTS PRO - VieNeu-TTS Setup
echo AMD GPU Acceleration via DirectML
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
if not exist "env" (
    python -m venv env
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

:: Activate virtual environment
echo [2/5] Activating virtual environment...
call env\Scripts\activate.bat

:: Upgrade pip
echo [3/5] Upgrading pip...
python -m pip install --upgrade pip

:: Install ONNX Runtime DirectML for AMD GPU
echo [4/5] Installing ONNX Runtime DirectML (AMD GPU support)...
pip install onnxruntime-directml>=1.19.0

:: Install VieNeu-TTS dependencies
echo [5/5] Installing VieNeu-TTS dependencies...
pip install torch>=2.3.0 torchaudio>=2.3.0
pip install transformers librosa soundfile "numpy<2.4"
pip install huggingface_hub pyyaml loguru rich addict accelerate
pip install onnx onnxsim neucodec phonemizer

:: Check for eSpeak NG
echo.
echo ============================================
echo Checking for eSpeak NG...
echo ============================================
if exist "C:\Program Files\eSpeak NG\libespeak-ng.dll" (
    echo [OK] eSpeak NG found.
) else (
    echo [WARNING] eSpeak NG NOT found at C:\Program Files\eSpeak NG\
    echo Vietnamese phonemization will fall back to dictionary-only mode.
    echo For best quality, please install eSpeak NG from:
    echo https://github.com/espeak-ng/espeak-ng/releases
)

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo To run the application:
echo   1. Activate: env\Scripts\activate
echo   2. Run: python -m app.main
echo.
pause
