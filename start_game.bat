@echo off
setlocal

set "CODE_DIR=%~dp0code"

echo [start] Launching SIGIL STRIKE ...
echo [start] Hand-sign detection runs automatically if model files are present.
echo [start] Keyboard input is always available.
echo [start] Tip: pass two team IDs to skip the bracket and play a single match,
echo [start]      e.g.   start_game.bat 1 4   (Team1 vs Team4)
echo.

python "%CODE_DIR%\main.py" %*
