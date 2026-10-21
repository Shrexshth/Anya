#!/usr/bin/env bash
# Anya Companion Startup Script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "  Starting Anya AI Voice Companion Setup & Launch "
echo "=================================================="

# Ensure .env exists
if [ ! -f .env ]; then
    echo "[!] .env not found. Copying .env.example to .env..."
    cp .env.example .env
fi

# Locate Python 3.11 if available, fallback to python3
PYTHON_BIN=""
if command -v python3.11 &>/dev/null; then
    PYTHON_BIN="python3.11"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    echo "[-] Error: Python 3 not found on system." >&2
    exit 1
fi

echo "[*] Using Python: $($PYTHON_BIN --version)"

# Check or create virtual environment
if [ ! -d ".venv" ]; then
    echo "[*] Creating virtual environment (.venv)..."
    "$PYTHON_BIN" -m venv .venv
fi

# Activate virtualenv
# shellcheck disable=SC1091
source .venv/bin/activate

# Check if dependencies need installation
if ! python -c "import PySide6, ollama, dotenv" &>/dev/null; then
    echo "[*] Installing/verifying pinned dependencies from requirements.txt..."
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
fi

echo "[*] Launching Anya application..."
exec python main.py "$@"
