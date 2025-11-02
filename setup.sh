#!/bin/bash

if [ -d "/opt/conda" ]; then
    echo "Conda environment detected"
    conda init
    conda activate pytorch
    export DATASETS_AUDIO_BACKEND=soundfile
elif [ -d "/opt/pytorch" ]; then
    echo "Python venv detected (PyTorch 2.6+)"
    source /opt/pytorch/bin/activate
else
    echo "Warning: No known Python environment found"
    echo "Looking for conda or venv..."
    if command -v conda &> /dev/null; then
        conda activate pytorch
    fi
fi

