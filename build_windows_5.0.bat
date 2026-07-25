@echo off
setlocal
cd /d "%~dp0"

echo [1/6] Checking Python...
python --version || goto :error

echo [2/6] Installing dependencies...
python -m pip install --upgrade pip || goto :error
python -m pip install -r requirements.txt || goto :error

echo [3/6] Checking release consistency...
python tools\check_release_consistency.py || goto :error

echo [4/6] Running tests...
python -m pytest || goto :error

echo [5/6] Building application...
python -m PyInstaller --noconfirm --clean DLTAnalyzerPro.spec || goto :error

echo [6/6] Done.
echo Application: dist\DLTAnalyzerPro\DLTAnalyzerPro.exe
echo To build the installer, install Inno Setup 6 and compile installer\DLTAnalyzerPro.iss
pause
exit /b 0

:error
echo.
echo Build failed. Review the error above.
pause
exit /b 1
