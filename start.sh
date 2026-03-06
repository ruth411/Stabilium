#!/bin/sh
PORT="${PORT:-8000}"
exec uvicorn api.main:app --host 0.0.0.0 --port "$PORT"
