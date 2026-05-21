@echo off
cd /d "%~dp0..\.."
python -m pytest code/test-cases/ -v
pause
