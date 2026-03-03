#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Nifty Options Prediction — Streamlit Web App Launcher
# Usage:
#   bash run_web.sh                   # default port 8501
#   bash run_web.sh --server.port 8502
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")"   # always run from project root

if [ ! -d ".venv" ]; then
    echo "ERROR: No .venv found. Run 'bash setup.sh' first."
    exit 1
fi

# Install streamlit if not present
if ! .venv/bin/python -c "import streamlit" 2>/dev/null; then
    echo "Installing streamlit..."
    .venv/bin/pip install "streamlit>=1.30.0" -q
fi

echo "─────────────────────────────────────────────────"
echo "  Nifty Options Prediction — Web App"
echo "  Open: http://localhost:8501"
echo "  Press Ctrl+C to stop."
echo "─────────────────────────────────────────────────"
echo ""

.venv/bin/streamlit run app.py \
    --server.headless true \
    --browser.gatherUsageStats false \
    "$@"
