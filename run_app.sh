#!/usr/bin/env bash
set -e

if [ -x "/tmp/freshscan_venv/bin/streamlit" ]; then
  STREAMLIT_BIN="/tmp/freshscan_venv/bin/streamlit"
elif [ -x ".venv/bin/streamlit" ]; then
  STREAMLIT_BIN=".venv/bin/streamlit"
elif command -v streamlit >/dev/null 2>&1; then
  STREAMLIT_BIN="$(command -v streamlit)"
else
  echo "[run_app] Installing Python dependencies into /tmp/freshscan_venv..."
  rm -rf /tmp/freshscan_venv
  python3 -m venv --without-pip --copies /tmp/freshscan_venv
  export UV_INSTALL_DIR=/tmp/uv_bin
  curl -LsSf https://astral.sh/uv/install.sh | sh
  /tmp/uv_bin/uv pip install --no-cache --python /tmp/freshscan_venv/bin/python -r requirements.txt
  STREAMLIT_BIN="/tmp/freshscan_venv/bin/streamlit"
fi

exec "$STREAMLIT_BIN" run app.py \
  --server.port "${PORT:-3000}" \
  --server.address "0.0.0.0" \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
