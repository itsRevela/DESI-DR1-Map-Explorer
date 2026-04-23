@echo off
setlocal enabledelayedexpansion
title DESI Map Explorer - Setup

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PY_VER=3.14.3"
set "PY_DIR=Python314"
set "PY_MIN_MAJOR=3"
set "PY_MIN_MINOR=13"

echo.
echo  ============================================
echo   DESI Map Explorer - Setup and Launch
echo  ============================================
echo.

:: --- Check for required files ---
if not exist "%SCRIPT_DIR%requirements.txt" (
    echo  [ERROR] requirements.txt not found.
    echo  Run this script from the project directory.
    pause
    exit /b 1
)

:: --- Check for Python ---
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo  [!] Python not found on your system.
    echo.

    where curl >nul 2>&1
    if %errorlevel% neq 0 (
        echo  [ERROR] curl is not available to download Python.
        echo  Please install Python %PY_VER% manually:
        echo  https://www.python.org/downloads/
        pause
        exit /b 1
    )

    echo  Downloading Python %PY_VER% installer...
    echo.
    curl -L --fail -o "%TEMP%\python-installer.exe" "https://www.python.org/ftp/python/%PY_VER%/python-%PY_VER%-amd64.exe"
    if %errorlevel% neq 0 (
        echo  [ERROR] Failed to download Python installer.
        echo  Please install Python %PY_VER% manually:
        echo  https://www.python.org/downloads/
        pause
        exit /b 1
    )

    echo.
    echo  Running Python installer...
    echo  IMPORTANT: Make sure "Add Python to PATH" is checked!
    echo.
    start /wait "" "%TEMP%\python-installer.exe" InstallAllUsers=0 PrependPath=1 Include_test=0
    if %errorlevel% neq 0 (
        echo  [ERROR] Python installer exited with an error.
        echo  If you cancelled, please re-run this script.
        del "%TEMP%\python-installer.exe" >nul 2>&1
        pause
        exit /b 1
    )
    del "%TEMP%\python-installer.exe" >nul 2>&1

    :: Refresh PATH for this session
    set "PATH=%LocalAppData%\Programs\Python\%PY_DIR%\;%LocalAppData%\Programs\Python\%PY_DIR%\Scripts\;%PATH%"

    where python >nul 2>&1
    if %errorlevel% neq 0 (
        echo.
        echo  [ERROR] Python still not found after installation.
        echo  Please close this window, reopen a new terminal,
        echo  and run this script again.
        pause
        exit /b 1
    )
)

:: --- Version check ---
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PYVER_FULL=%%v"
for /f "tokens=1,2 delims=." %%a in ("!PYVER_FULL!") do (
    set "PYMAJ=%%a"
    set "PYMIN=%%b"
)
echo  [OK] Python !PYVER_FULL! detected

if !PYMAJ! lss %PY_MIN_MAJOR% (
    echo  [ERROR] Python %PY_MIN_MAJOR%.%PY_MIN_MINOR%+ is required. Found: !PYVER_FULL!
    pause
    exit /b 1
)
if !PYMAJ! equ %PY_MIN_MAJOR% if !PYMIN! lss %PY_MIN_MINOR% (
    echo  [ERROR] Python %PY_MIN_MAJOR%.%PY_MIN_MINOR%+ is required. Found: !PYVER_FULL!
    pause
    exit /b 1
)
echo.

:: --- Create or verify venv ---
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo  [!] Existing venv is broken. Recreating...
        rmdir /s /q venv
    )
)

if not exist "venv\Scripts\python.exe" (
    echo  [..] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created
    echo.

    echo  [..] Installing dependencies (this may take a minute)...
    venv\Scripts\pip install --upgrade pip >nul 2>&1
    venv\Scripts\pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo  [ERROR] Failed to install dependencies. Cleaning up...
        rmdir /s /q venv
        pause
        exit /b 1
    )
    echo.
    echo  [OK] All dependencies installed
) else (
    echo  [OK] Virtual environment ready
)
echo.

:: --- Dataset selection ---
echo  ============================================
echo   Choose a dataset:
echo  ============================================
echo.
echo   1) EDR  - Early Data Release
echo            ~2 GB download, ~1.2 million objects
echo            Smaller dataset, good for testing
echo            or lower-end hardware
echo.
echo   2) DR1  - Data Release 1 (Full Survey)
echo            ~21 GB download, ~18 million objects
echo            Complete dataset, requires decent
echo            GPU and ~4 GB RAM
echo.

set "choice="
set /p choice="  Enter choice (1 or 2): "

if "!choice!"=="1" (
    set "DATASET=edr"
    echo.
    echo  Starting with EDR dataset...
) else if "!choice!"=="2" (
    set "DATASET=dr1"
    echo.
    echo  Starting with DR1 dataset...
) else (
    echo.
    echo  Invalid choice, defaulting to EDR...
    set "DATASET=edr"
)

echo.
echo  First launch will download the FITS catalog.
echo  This may take a while depending on your connection.
echo  The download supports resume if interrupted.
echo.

venv\Scripts\python main.py --dataset "!DATASET!"

echo.
pause
