#!/usr/bin/env bash
# Smoke test: start the server against a temp project root and hit the main endpoints.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT=$(mktemp -d)
PORT="${PORT:-8099}"
PY=".venv3.13/bin/python"
[[ -x "$PY" ]] || PY="python3"

"$PY" -m imagagen --mock --root "$ROOT" --port "$PORT" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# wait for the server
for _ in $(seq 1 50); do
  if curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then break; fi
  sleep 0.2
done

echo "--- health"
curl -sf "http://127.0.0.1:$PORT/api/health"; echo
echo "--- generate"
curl -sf -X POST "http://127.0.0.1:$PORT/api/generate" \
  -H 'content-type: application/json' \
  -d '{"prompt":"smoke test turtle","width":64,"height":64,"steps":4,"seed":7}'; echo
echo "--- output dir"
ls "$ROOT/output"
echo "SMOKE OK"
