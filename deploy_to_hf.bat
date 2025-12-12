@echo off
REM ============================================
REM Multi-Disease Detection - Deployment Script
REM ============================================
REM Run this script to prepare files for Hugging Face Spaces deployment

echo.
echo ========================================
echo Multi-Disease Detection Deployment Setup
echo ========================================
echo.

set HF_SPACE_DIR=hf-space-deploy

REM Create deployment directory
if exist %HF_SPACE_DIR% (
    echo Cleaning existing deployment directory...
    rmdir /s /q %HF_SPACE_DIR%
)
mkdir %HF_SPACE_DIR%

echo [1/6] Copying Dockerfile and requirements...
copy deployment\huggingface\Dockerfile %HF_SPACE_DIR%\
copy deployment\huggingface\requirements.txt %HF_SPACE_DIR%\
copy deployment\huggingface\README.md %HF_SPACE_DIR%\
copy deployment\huggingface\.gitattributes %HF_SPACE_DIR%\

echo [2/6] Copying source code...
xcopy /E /I /Y src %HF_SPACE_DIR%\src

echo [3/6] Copying configs...
xcopy /E /I /Y configs %HF_SPACE_DIR%\configs

echo [4/6] Copying models (this may take a while)...
xcopy /E /I /Y models %HF_SPACE_DIR%\models

echo [5/6] Initializing Git repository...
cd %HF_SPACE_DIR%
git init
git lfs install
git lfs track "*.h5"
git lfs track "*.weights.h5"
git add .gitattributes

echo [6/6] Adding all files...
git add .
git commit -m "Initial deployment"

echo.
echo ========================================
echo Deployment package ready!
echo ========================================
echo.
echo Next steps:
echo 1. Create a Space on huggingface.co/spaces
echo 2. Run: git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
echo 3. Run: git push -u origin main
echo.
echo Your deployment folder: %HF_SPACE_DIR%
echo.

cd ..
pause
