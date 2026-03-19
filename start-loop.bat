@echo off
title CCDB Auto-Restart
cd /d C:\Users\konum\git\tech-forward\ebibibi-discord-bridge

:loop
echo [%date% %time%] Starting ccdb...
uv run ccdb start
echo [%date% %time%] ccdb stopped (exit code: %errorlevel%). Restarting in 5 seconds...
timeout /t 5 /nobreak >nul
goto loop
