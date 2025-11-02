#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Voice Lab Environment Setup ==="
echo ""

if [ -d "/opt/conda" ]; then
    echo "Conda environment detected"
    conda init
    conda activate pytorch
    export DATASETS_AUDIO_BACKEND=soundfile
    echo "✓ Conda environment activated: pytorch"
elif [ -d "/opt/pytorch" ]; then
    echo "Python venv detected (PyTorch 2.6+)"
    source /opt/pytorch/bin/activate
    echo "✓ Virtual environment activated: /opt/pytorch"
else
    echo "Warning: No known Python environment found"
    echo "Looking for conda or venv..."
    if command -v conda &> /dev/null; then
        conda activate pytorch
        echo "✓ Conda environment activated: pytorch"
    else
        echo "Error: No Python environment found"
        exit 1
    fi
fi

if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo ""
    echo "Installing packages from requirements.txt..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
    echo "✓ Packages installed"
else
    echo "Warning: requirements.txt not found in $SCRIPT_DIR"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Environment is now active in this shell session."
echo ""
echo "To activate in future sessions, add to ~/.bashrc:"
if [ -d "/opt/conda" ]; then
    echo "  conda activate pytorch"
elif [ -d "/opt/pytorch" ]; then
    echo "  source /opt/pytorch/bin/activate"
fi
echo ""

