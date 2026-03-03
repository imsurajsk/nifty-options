#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Nifty Options Prediction System — One-time Setup Script
# Run: bash setup.sh
# ──────────────────────────────────────────────────────────────────────────────

set -e

echo ""
echo "========================================"
echo "  Nifty Options Prediction System Setup"
echo "========================================"
echo ""

# 1. Create virtual environment
echo "[1/3] Creating virtual environment..."
python3 -m venv .venv
echo "      ✓ Created .venv/"

# 2. Install dependencies
echo "[2/3] Installing dependencies..."
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q
echo "      ✓ All packages installed"

# 3. Smoke test
echo "[3/3] Running quick smoke test..."
.venv/bin/python3 -c "
import yfinance, pandas, numpy, requests, rich, scipy
print('      ✓ All imports successful')
"

echo ""
echo "========================================"
echo "  Setup complete!"
echo ""
echo "  To run the prediction system:"
echo ""
echo "    source .venv/bin/activate"
echo "    python main.py"
echo ""
echo "  Or without activating venv:"
echo ""
echo "    .venv/bin/python main.py"
echo ""
echo "  To override capital (default is ₹1,00,000):"
echo "    python main.py --capital 500000"
echo "========================================"
echo ""
