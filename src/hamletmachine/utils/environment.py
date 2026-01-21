"""
Environment detection utilities for local vs cloud execution.
"""

import os
from pathlib import Path
from typing import Optional


def is_colab() -> bool:
    """Check if running on Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def is_kaggle() -> bool:
    """Check if running on Kaggle."""
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def is_cloud() -> bool:
    """Check if running on any cloud platform."""
    return is_colab() or is_kaggle()


def get_base_path() -> Path:
    """
    Get the base path for the project.
    Returns appropriate path for Colab, Kaggle, or local execution.
    """
    if is_colab():
        # Colab default working directory
        base = Path("/content")
        # Check if project is in /content/hamletmachine
        if (base / "hamletmachine").exists():
            return base / "hamletmachine"
        return base
    elif is_kaggle():
        # Kaggle default working directory
        return Path("/kaggle/working")
    else:
        # Local execution - assume we're in the project root
        # This will work if script is run from project root
        current = Path.cwd()
        # Try to find project root (look for pyproject.toml or setup.py)
        while current != current.parent:
            if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
                return current
            current = current.parent
        return Path.cwd()


def get_data_dir() -> Path:
    """Get the data directory path."""
    base = get_base_path()
    return base / "data"


def get_processed_data_dir() -> Path:
    """Get the processed data directory path."""
    return get_data_dir() / "processed"


def get_models_dir() -> Path:
    """Get the models directory path."""
    base = get_base_path()
    return base / "models"


def get_checkpoints_dir() -> Optional[Path]:
    """
    Get the checkpoints directory path.
    On Colab, prefers Google Drive if mounted.
    """
    if is_colab():
        # Check if Google Drive is mounted
        drive_path = Path("/content/drive/MyDrive/hamletmachine/checkpoints")
        if drive_path.exists() or Path("/content/drive").exists():
            drive_path.parent.mkdir(parents=True, exist_ok=True)
            return drive_path
        # Fall back to local Colab storage
        return Path("/content/models/checkpoints")
    else:
        return get_models_dir() / "checkpoints"


def get_logs_dir() -> Path:
    """Get the logs directory path."""
    base = get_base_path()
    return base / "logs"


def setup_colab_directories():
    """Create necessary directories on Colab."""
    if not is_colab():
        return
    
    dirs = [
        get_processed_data_dir(),
        get_checkpoints_dir(),
        get_logs_dir(),
    ]
    
    for dir_path in dirs:
        if dir_path:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")


def get_gpu_info() -> dict:
    """Get GPU information if available."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "device_count": torch.cuda.device_count(),
            }
        else:
            return {"available": False}
    except ImportError:
        return {"available": False, "error": "PyTorch not installed"}


def print_environment_info():
    """Print information about the current environment."""
    print("Environment Information:")
    print(f"  Platform: {'Colab' if is_colab() else 'Kaggle' if is_kaggle() else 'Local'}")
    print(f"  Base path: {get_base_path()}")
    print(f"  Data dir: {get_processed_data_dir()}")
    print(f"  Checkpoints dir: {get_checkpoints_dir()}")
    print(f"  Logs dir: {get_logs_dir()}")
    
    gpu_info = get_gpu_info()
    if gpu_info.get("available"):
        print(f"  GPU: {gpu_info['device_name']}")
        print(f"  GPU Memory: {gpu_info['memory_gb']:.2f} GB")
    else:
        print("  GPU: Not available")


if __name__ == "__main__":
    print_environment_info()
