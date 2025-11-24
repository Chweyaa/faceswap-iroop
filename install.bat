@echo off
echo ============================================
echo faceswap-iroop Installation Script
echo ============================================
echo.

echo Step 1: Creating virtual environment...
uv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Make sure uv is installed: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    pause
    exit /b 1
)
echo.

echo Step 2: Activating virtual environment...
call .venv\Scripts\activate.bat
echo.

echo Step 3: Installing base requirements...
uv pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install base requirements
    pause
    exit /b 1
)
echo.

echo Step 4: Installing TensorFlow...
uv pip install tensorflow==2.12.1
if errorlevel 1 (
    echo ERROR: Failed to install TensorFlow
    pause
    exit /b 1
)
echo.

echo Step 5: Installing typing-extensions...
uv pip install typing-extensions==4.15.0
if errorlevel 1 (
    echo ERROR: Failed to install typing-extensions
    pause
    exit /b 1
)
echo.

echo ============================================
echo Installation completed successfully!
echo ============================================
echo.
echo Next steps:
echo 1. Download required models and place them in the 'models' folder:
echo    - GFPGANv1.4.pth
echo    - inswapper_128_fp16.onnx
echo.
echo 2. Run the application:
echo    - For GPU: run-cuda.bat
echo    - For CPU: run-cpu.bat
echo.
pause
