#!/bin/bash
# Setup script for Google Colab
# This script installs dependencies and sets up the environment for training

set -e  # Exit on error

echo "🚀 Setting up hamletmachine training environment on Colab..."

# Check if running on Colab
if [ -z "$COLAB_GPU" ] && ! python -c "import google.colab" 2>/dev/null; then
    echo "⚠️  Warning: This script is designed for Google Colab"
    echo "   It will still work, but some features may not be available"
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q --upgrade pip
pip install -q transformers>=4.35.0 datasets>=2.14.0 accelerate>=0.24.0 tokenizers>=0.15.0
pip install -q torch>=2.1.0 numpy>=1.24.0 pandas>=2.0.0 pyyaml>=6.0
pip install -q tensorboard>=2.15.0 tqdm>=4.66.0

# Optional: Install WandB
# pip install -q wandb

echo "✅ Dependencies installed!"

# Check GPU
echo "🔍 Checking GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU available: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('⚠️  No GPU detected! Enable GPU: Runtime → Change runtime type → GPU')
"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models/checkpoints
mkdir -p logs
mkdir -p data/processed

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Mount Google Drive (optional): from google.colab import drive; drive.mount('/content/drive')"
echo "2. Upload your processed dataset to data/processed/"
echo "3. Run the training notebook or script"
