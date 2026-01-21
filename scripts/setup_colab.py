#!/usr/bin/env python3
"""
Setup script for Google Colab environment.
Installs dependencies and verifies GPU availability.
"""

import subprocess
import sys
import os


def run_command(cmd, check=True):
    """Run a shell command and print output."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result


def check_colab():
    """Check if running on Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU available: {device_name}")
            print(f"   Memory: {memory_gb:.2f} GB")
            return True
        else:
            print("⚠️  No GPU detected!")
            print("   Enable GPU: Runtime → Change runtime type → GPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet (will be installed)")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    packages = [
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "tokenizers>=0.15.0",
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tensorboard>=2.15.0",
        "tqdm>=4.66.0",
    ]
    
    for package in packages:
        run_command(f"pip install -q {package}", check=False)
    
    print("✅ Dependencies installed!")


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    dirs = [
        "models/checkpoints",
        "logs",
        "data/processed",
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"   Created: {dir_path}")
    
    print("✅ Directories created!")


def main():
    """Main setup function."""
    print("🚀 Setting up hamletmachine training environment...")
    print()
    
    # Check if on Colab
    is_colab = check_colab()
    if is_colab:
        print("✅ Running on Google Colab")
    else:
        print("⚠️  Not running on Colab (this is fine for local setup)")
    print()
    
    # Install dependencies
    install_dependencies()
    print()
    
    # Check GPU (after torch is installed)
    check_gpu()
    print()
    
    # Create directories
    create_directories()
    print()
    
    print("✅ Setup complete!")
    print()
    print("Next steps:")
    if is_colab:
        print("1. Mount Google Drive (optional):")
        print("   from google.colab import drive")
        print("   drive.mount('/content/drive')")
    print("2. Upload your processed dataset to data/processed/")
    print("3. Run the training notebook or script")


if __name__ == "__main__":
    main()
