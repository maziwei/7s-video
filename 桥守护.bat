@echo off
cd /d "%~dp0"
:loop
netstat -ano | findstr /C:"127.0.0.1:8677" | findstr LISTENING >nul
if %errorlevel%==0 goto sleep
set PYTHONUTF8=1
py -3 --version >nul 2>&1
if %errorlevel%==0 goto startpy
python --version >nul 2>&1
if %errorlevel%==0 goto startpython
timeout /t 60 /nobreak >nul
goto loop
:startpy
start "Bridge-8677" /min py -3 comfy_bridge.py
goto sleep
:startpython
start "Bridge-8677" /min python comfy_bridge.py
:sleep
timeout /t 15 /nobreak >nul
goto loop
