#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Voice Lab GPU Setup ==="
echo ""

if [ -d "/opt/pytorch" ]; then
    echo "Detected: AWS Deep Learning AMI (venv)"
    source /opt/pytorch/bin/activate
    echo "✓ Activated /opt/pytorch"
elif [ -d "/opt/conda" ]; then
    echo "Detected: Conda environment"
    source /opt/conda/etc/profile.d/conda.sh
    conda activate pytorch
    echo "✓ Activated conda pytorch"
else
    echo "No known environment found. Using system Python."
fi

echo ""
echo "Python: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
echo ""

echo "Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq ffmpeg
    echo "✓ FFmpeg installed"
else
    echo "⚠ apt-get not found, skipping ffmpeg install"
fi

echo ""
echo "Installing Python packages..."
pip install -q -r "$PROJECT_DIR/requirements.txt"
echo "✓ Python packages installed"

echo ""
echo "Verifying installation..."
python -c "
import torch
import transformers
import datasets
import wandb
import hydra
print('✓ All packages imported successfully')
print(f'  PyTorch: {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== WandB Setup ==="
if [ -f ~/.netrc ] && grep -q "api.wandb.ai" ~/.netrc; then
    echo "✓ WandB already configured"
else
    echo "WandB not configured. Run: wandb login"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To train:"
echo "  cd $PROJECT_DIR"
echo "  python encoder/train.py"
echo ""
echo "To train with overrides:"
echo "  python encoder/train.py batch_size=8 learning_rate=5e-5"
echo ""

