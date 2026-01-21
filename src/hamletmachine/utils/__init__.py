"""Shared utilities and helpers."""

from .environment import (
    is_colab,
    is_kaggle,
    is_cloud,
    get_base_path,
    get_data_dir,
    get_processed_data_dir,
    get_models_dir,
    get_checkpoints_dir,
    get_logs_dir,
    setup_colab_directories,
    get_gpu_info,
    print_environment_info,
)

__all__ = [
    "is_colab",
    "is_kaggle",
    "is_cloud",
    "get_base_path",
    "get_data_dir",
    "get_processed_data_dir",
    "get_models_dir",
    "get_checkpoints_dir",
    "get_logs_dir",
    "setup_colab_directories",
    "get_gpu_info",
    "print_environment_info",
]
