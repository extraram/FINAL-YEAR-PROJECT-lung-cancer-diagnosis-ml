@echo off
REM Lung Cancer Diagnosis AI - Quick Launch Script
REM Sets up the environment and launches the Flask + HTML/CSS/JS web app

echo.
echo ========================================
echo   LUNG CANCER DIAGNOSIS AI
echo   Medical Imaging Deep Learning System
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if dependencies are installed
pip show flask >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
    pip install flask flask-cors
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Check if models exist
if not exist "models\cnn_lung_cancer_model.h5" (
    echo.
    echo [WARNING] Trained models not found!
    echo.
    echo Please run the training script first:
    echo   python train.py
    echo.
    echo After training completes, run this script again.
    pause
    exit /b 1
)

REM Launch the server
echo.
echo [INFO] Starting Lung Cancer Diagnosis AI...
echo [INFO] Opening at http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

python server.py
