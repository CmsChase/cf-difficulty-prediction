@echo off
setlocal
cd /d "%~dp0\.."
python scripts\statement_fetch_gui.py
endlocal
