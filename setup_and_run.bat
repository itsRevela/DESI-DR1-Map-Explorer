@echo off
setlocal enabledelayedexpansion
title DESI Map Explorer - Setup

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PY_VER=3.14.3"
set "PY_DIR=Python314"

echo.
echo  ============================================
echo   DESI Map Explorer - Setup and Launch
echo  ============================================
echo.

:: --- Check required files ---
if not exist "%SCRIPT_DIR%requirements.txt" goto :no_requirements

:: --- Check for Python ---
where python >nul 2>&1
if !errorlevel! neq 0 goto :install_python
goto :check_version

:install_python
echo  Python not found on your system.
echo.
where curl >nul 2>&1
if !errorlevel! neq 0 goto :no_curl
echo  Downloading Python %PY_VER% installer...
echo.
curl -L --fail -o "%TEMP%\python-installer.exe" "https://www.python.org/ftp/python/%PY_VER%/python-%PY_VER%-amd64.exe"
if !errorlevel! neq 0 goto :download_failed
echo.
echo  Running Python installer...
echo  IMPORTANT: Make sure "Add Python to PATH" is checked!
echo.
start /wait "" "%TEMP%\python-installer.exe" InstallAllUsers=0 PrependPath=1 Include_test=0
del "%TEMP%\python-installer.exe" >nul 2>&1
set "PATH=%LocalAppData%\Programs\Python\%PY_DIR%\;%LocalAppData%\Programs\Python\%PY_DIR%\Scripts\;%PATH%"
where python >nul 2>&1
if !errorlevel! neq 0 goto :python_not_found_after
goto :check_version

:check_version
set "PYVER_FULL=unknown"
set "PYMAJ=0"
set "PYMIN=0"
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PYVER_FULL=%%v"
for /f "tokens=1,2 delims=." %%a in ("!PYVER_FULL!") do (
    set "PYMAJ=%%a"
    set "PYMIN=%%b"
)
echo  [OK] Python !PYVER_FULL! detected
if !PYMAJ! equ 0 goto :bad_version
if !PYMAJ! lss 3 goto :bad_version
if !PYMAJ! equ 3 if !PYMIN! lss 13 goto :bad_version
echo.

:: --- Check venv ---
if not exist "venv\Scripts\python.exe" goto :create_venv
venv\Scripts\python --version >nul 2>&1
if !errorlevel! neq 0 goto :recreate_venv
echo  [OK] Virtual environment ready
echo.
goto :choose_dataset

:recreate_venv
echo  Existing venv is broken. Recreating...
rmdir /s /q venv

:create_venv
echo  Creating virtual environment...
python -m venv venv
if !errorlevel! neq 0 goto :venv_failed
echo  [OK] Virtual environment created
echo.
echo  Installing dependencies (this may take a minute)...
venv\Scripts\pip install --upgrade pip >nul 2>&1
venv\Scripts\pip install -r requirements.txt
if !errorlevel! neq 0 goto :deps_failed
echo.
echo  [OK] All dependencies installed
echo.

:: --- Dataset selection ---
:choose_dataset
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

if "!choice!"=="1" set "DATASET=edr" & goto :launch
if "!choice!"=="2" set "DATASET=dr1" & goto :launch
echo  Invalid choice, defaulting to EDR...
set "DATASET=edr"

:launch
echo.
echo  First launch will download the FITS catalog.
echo  This may take a while depending on your connection.
echo  The download supports resume if interrupted.
echo.
venv\Scripts\python main.py --dataset "!DATASET!"
echo.
pause
exit /b 0

:: --- Error handlers ---
:no_requirements
echo  [ERROR] requirements.txt not found.
echo  Run this script from the project directory.
pause
exit /b 1

:no_curl
echo  [ERROR] curl is not available to download Python.
echo  Please install Python %PY_VER% manually:
echo  https://www.python.org/downloads/
pause
exit /b 1

:download_failed
echo  [ERROR] Failed to download Python installer.
echo  Please install Python %PY_VER% manually:
echo  https://www.python.org/downloads/
pause
exit /b 1

:python_not_found_after
echo.
echo  [ERROR] Python still not found after installation.
echo  Close this window, reopen a new terminal, and try again.
pause
exit /b 1

:bad_version
echo  [ERROR] Python 3.13 or higher is required. Found: !PYVER_FULL!
pause
exit /b 1

:venv_failed
echo  [ERROR] Failed to create virtual environment.
pause
exit /b 1

:deps_failed
echo.
echo  [ERROR] Failed to install dependencies. Cleaning up...
rmdir /s /q venv
pause
exit /b 1

