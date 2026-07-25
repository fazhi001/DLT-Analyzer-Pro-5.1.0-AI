@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONPATH=%CD%\src
py -3.12 main.py
if errorlevel 1 pause
