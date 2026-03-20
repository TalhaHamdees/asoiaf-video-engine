@echo off
setlocal
title ASOIAF Video Engine
cd /d "%~dp0"

:: Auto-update from GitHub
git --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Checking for updates...
    git pull origin main 2>nul
    if %errorlevel% neq 0 (
        echo [WARN] Could not pull updates. Continuing with current version.
    ) else (
        echo [OK] Up to date.
    )
) else (
    echo [WARN] Git not found, skipping auto-update.
)

:: Activate venv
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

:: Check if requirements changed (compare hash)
set HASH_FILE=temp\.requirements_hash
if not exist "temp" mkdir temp

certutil -hashfile requirements.txt MD5 2>nul | findstr /v ":" > "%HASH_FILE%.new" 2>nul
if exist "%HASH_FILE%" (
    fc /b "%HASH_FILE%" "%HASH_FILE%.new" >nul 2>&1
    if %errorlevel% neq 0 (
        echo Dependencies changed, updating...
        pip install -r requirements.txt
        copy /y "%HASH_FILE%.new" "%HASH_FILE%" >nul
    )
) else (
    copy /y "%HASH_FILE%.new" "%HASH_FILE%" >nul
)
del "%HASH_FILE%.new" 2>nul

:: Launch the app
echo.
echo Starting ASOIAF Video Engine...
echo (Close this window to stop the app)
echo.
python app.py
