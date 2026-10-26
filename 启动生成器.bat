@echo off
cd /d "%~dp0"

netstat -ano | findstr /C:"127.0.0.1:8677" | findstr LISTENING >nul
if %errorlevel%==0 goto open

set PYTHONUTF8=1
py -3 --version >nul 2>&1
if %errorlevel%==0 goto usepy
python --version >nul 2>&1
if %errorlevel%==0 goto usepython
echo [WARN] Python not found. Feishu/BGM/random-bg unavailable.
echo        Install Python or run start_bridge.ps1 for full features.
goto open

:usepy
start "Bridge-8677" /min py -3 comfy_bridge.py
goto waitbridge

:usepython
start "Bridge-8677" /min python comfy_bridge.py

:waitbridge
for /l %%i in (1,1,30) do (
  netstat -ano | findstr /C:"127.0.0.1:8677" | findstr LISTENING >nul
  if not errorlevel 1 goto open
  timeout /t 1 /nobreak >nul
)

:open
start "" "榜单视频生成器.html"