#!/bin/bash

set -e

echo "=== Voice Lab Setup Script ==="
echo "Setting up environment for PyTorch 2.8 + torchcodec AudioDecoder support"
echo ""

if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run with sudo"
    echo "Usage: sudo ./setup_current_gpu.sh"
    exit 1
fi

if [ ! -d "/opt/pytorch" ]; then
    echo "Error: /opt/pytorch venv not found. This script is for AWS DL AMI PyTorch 2.6+"
    exit 1
fi

echo "Installing system dependencies..."
echo "  - FFmpeg (required for torchcodec)"
apt-get update -qq
apt-get install -y -qq ffmpeg

echo ""
echo "Activating Python virtual environment..."
source /opt/pytorch/bin/activate

echo ""
echo "Current environment:"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
echo ""

echo "Installing Python packages..."
echo "  - accelerate"
pip install -q accelerate==1.11.0

echo "  - transformers"
pip install -q transformers==4.57.1

echo "  - datasets"
pip install -q datasets

echo "  - librosa"
pip install -q librosa

echo "  - soundfile"
pip install -q soundfile

echo "  - torchcodec (with AudioDecoder support)"
pip install -q torchcodec==0.7.0

echo ""
echo "Verifying installations..."
python -c "
import torch
import transformers
import datasets
import librosa
import soundfile
from torchcodec.decoders import AudioDecoder
print('✓ All packages imported successfully')
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ Transformers: {transformers.__version__}')
print(f'✓ Datasets: {datasets.__version__}')
print(f'✓ torchcodec AudioDecoder: Available')
"

echo ""
echo "Checking FFmpeg installation..."
ffmpeg -version | head -1

echo ""
echo "=== Setup Complete ==="
echo ""
echo "IMPORTANT: This AMI uses Python venv (not conda)"
echo ""
echo "To activate environment in new shell sessions:"
echo "  source /opt/pytorch/bin/activate"
echo ""
echo "Or add to your ~/.bashrc:"
echo "  echo 'source /opt/pytorch/bin/activate' >> ~/.bashrc"
echo ""
echo "To run training:"
echo "  cd /home/ubuntu/voice_lab"
echo "  python encoder/train.py"
echo ""
echo "Quick reference:"
echo "  Install package:  pip install <package>"
echo "  List packages:    pip list"
echo "  Deactivate:       deactivate"

