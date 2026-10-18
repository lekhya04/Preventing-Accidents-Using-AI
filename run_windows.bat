@echo off
title SafeDrive AI
cd /d "%~dp0"
echo Installing requirements (first time only takes a minute)...
python -m pip install -r requirements.txt
echo.
echo Starting SafeDrive AI - the browser opens automatically at http://localhost:8000
echo (if the saved models do not match your Python packages, it retrains first: about 5-10 minutes)
python src\server.py
pause
