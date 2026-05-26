@echo off
REM Lung Cancer Diagnosis AI - Training Script
REM This batch file trains the CNN and ResNet50 models

echo.
echo ========================================
echo   LUNG CANCER DIAGNOSIS AI
echo   Model Training Pipeline
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
pip show tensorflow >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies (this may take a while)...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Check dataset
if not exist "lung cancer dataset" (
    echo [ERROR] Dataset folder not found!
    echo Please ensure the "lung cancer dataset" folder exists in the project root
    pause
    exit /b 1
)

REM Run training
echo.
echo [INFO] Starting model training...
echo [INFO] This will train CNN and ResNet50 models
echo.
echo Dataset Information:
echo  - Normal cases: lung cancer dataset\Normal cases
echo  - Benign cases: lung cancer dataset\Benign cases
echo  - Malignant cases: lung cancer dataset\Malignant cases
echo.
echo Training Configuration:
echo  - Train/Val/Test split: 70%% / 15%% / 15%%
echo  - Image size: 224x224
echo  - Batch size: 32
echo  - Early stopping: Enabled
echo.
echo This may take 30-60 minutes depending on your hardware.
echo Models will be saved to: models\
echo Reports will be saved to: reports\
echo.
echo Press any key to start training...
pause >nul

python train.py

if errorlevel 1 (
    echo.
    echo [ERROR] Training failed!
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Training completed!
echo Models saved to: models\
echo Reports saved to: reports\
echo.
echo You can now run the application with: run_app.bat
echo.
pause
