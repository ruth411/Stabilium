#!/bin/sh
PORT_VALUE="${PORT:-8000}"
case "$PORT_VALUE" in
  ''|*[!0-9]*)
    PORT_VALUE="8000"
    ;;
esac

exec uvicorn api.main:app --host 0.0.0.0 --port "$PORT_VALUE"
