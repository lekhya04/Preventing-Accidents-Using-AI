#!/usr/bin/env bash
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
echo "Starting SafeDrive AI at http://localhost:8000 (retrains first if saved models don't load)"
python3 src/server.py
