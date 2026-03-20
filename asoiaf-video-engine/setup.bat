@echo off
setlocal enabledelayedexpansion
title ASOIAF Video Engine — Setup
echo ============================================================
echo   ASOIAF Video Engine — One-Time Setup
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found on PATH.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Check Python version >= 3.10
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)
if %PYMAJOR% lss 3 (
    echo [ERROR] Python 3.10+ required, found %PYVER%
    pause
    exit /b 1
)
if %PYMAJOR%==3 if %PYMINOR% lss 10 (
    echo [ERROR] Python 3.10+ required, found %PYVER%
    pause
    exit /b 1
)
echo [OK] Python %PYVER% found.

:: Check Git
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] Git not found on PATH. Auto-updates will not work.
    echo You can install Git from https://git-scm.com/download/win
    echo.
) else (
    echo [OK] Git found.
)

:: Create virtual environment
echo.
echo Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)

:: Activate and install dependencies
echo.
echo Installing dependencies (this may take a few minutes)...
call venv\Scripts\activate.bat
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo [OK] Dependencies installed.

:: Pre-download Whisper model
echo.
echo Pre-downloading Whisper model (base)...
python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu')" 2>nul
if %errorlevel% equ 0 (
    echo [OK] Whisper model ready.
) else (
    echo [WARN] Whisper model download may have failed. It will retry on first use.
)

:: Setup .env file
echo.
if not exist ".env" (
    echo Setting up API keys...
    copy .env.example .env >nul

    set /p ELEVENLABS_KEY="Enter your ElevenLabs API Key (or press Enter to skip): "
    if not "!ELEVENLABS_KEY!"=="" (
        powershell -Command "(Get-Content .env) -replace 'your_elevenlabs_api_key_here', '!ELEVENLABS_KEY!' | Set-Content .env"
    )

    set /p ANTHROPIC_KEY="Enter your Anthropic API Key (or press Enter to skip): "
    if not "!ANTHROPIC_KEY!"=="" (
        powershell -Command "(Get-Content .env) -replace 'your_anthropic_api_key_here', '!ANTHROPIC_KEY!' | Set-Content .env"
    )

    echo [OK] .env file created. You can edit it later in Settings.
) else (
    echo [OK] .env file already exists.
)

:: Create desktop shortcut
echo.
echo Creating desktop shortcut...
set SCRIPT_DIR=%~dp0
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\ASOIAF Video Engine.lnk'); $s.TargetPath = '%SCRIPT_DIR%launch.bat'; $s.WorkingDirectory = '%SCRIPT_DIR%'; $s.IconLocation = 'shell32.dll,23'; $s.Description = 'Launch ASOIAF Video Engine'; $s.Save()"
echo [OK] Desktop shortcut created.

:: Done
echo.
echo ============================================================
echo   Setup complete! Double-click "ASOIAF Video Engine" on
echo   your desktop to launch, or run launch.bat directly.
echo ============================================================
echo.
pause
