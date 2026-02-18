#!/usr/bin/env bash
# setup_env.sh: install pip, requirements, and editable Utils package into a given venv
# Usage: ./scripts/setup_env.sh <venv_path> <requirements_file_path>

# Exit immediately if a command exits with a non-zero status
set -e

venv_path="$1"
requirements_file="$2"

# ------------------------------------------------------------------
# 1. Configurazione TMPDIR (Utile per GitHub Actions o server pieni)
# ------------------------------------------------------------------
if [ -n "$TMPDIR" ]; then
    export TMPDIR TMP TEMP
    # Set PIP_CACHE_DIR inside TMPDIR to avoid default cache location permissions issues
    export PIP_CACHE_DIR="$TMPDIR/pip_cache"
    mkdir -p "$PIP_CACHE_DIR"
    echo "  → Using TMPDIR and PIP_CACHE_DIR=$PIP_CACHE_DIR"
fi

# ------------------------------------------------------------------
# 2. Controllo esistenza pip nel venv
# ------------------------------------------------------------------
if [ ! -x "${venv_path}/bin/pip" ]; then
    echo "❌ Error: pip not found in venv at ${venv_path}/bin/pip"
    exit 1
fi

# ------------------------------------------------------------------
# 3. Aggiornamento pip/setuptools
# ------------------------------------------------------------------
echo "  → Upgrading pip, setuptools, wheel in venv..."
if [ "$NO_CACHE" = "1" ]; then
    "${venv_path}/bin/pip" install --upgrade pip setuptools wheel --no-cache-dir
else
    "${venv_path}/bin/pip" install --upgrade pip setuptools wheel
fi

# ------------------------------------------------------------------
# 4. Installazione Requirements
# ------------------------------------------------------------------
if [ -f "$requirements_file" ]; then
    echo "  → Installing from $requirements_file"
    if [ "$NO_CACHE" = "1" ]; then
        "${venv_path}/bin/pip" install --no-cache-dir -r "$requirements_file"
    else
        "${venv_path}/bin/pip" install -r "$requirements_file"
    fi
else
    echo "  → No requirements file at $requirements_file; skipping dependencies."
fi

# ------------------------------------------------------------------
# 5. Installazione Utils (il tuo pacchetto locale)
# ------------------------------------------------------------------
echo "  → Installing shared Utils package (editable) into venv"

# Controllo di sicurezza: siamo nella root?
if [ ! -f "setup.py" ]; then
    echo "❌ Error: setup.py not found in current directory. Cannot install Utils."
    echo "   Ensure you run this script from the project root."
    exit 1
fi

if [ "$NO_CACHE" = "1" ]; then
    "${venv_path}/bin/pip" install -e . --no-cache-dir
else
    "${venv_path}/bin/pip" install -e .
fi

echo "✅ Environment setup complete."