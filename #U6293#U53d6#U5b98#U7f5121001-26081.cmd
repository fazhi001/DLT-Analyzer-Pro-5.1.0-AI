@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3.12 tools\fetch_official_21001_26081.py
pause
