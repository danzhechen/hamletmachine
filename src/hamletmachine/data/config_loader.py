"""
Configuration loading utilities for data processing.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_data_config(config_path: Path) -> Dict[str, Any]:
    """
    Load data processing configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration values
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    
    # Validate required sections
    required_sections = ['input', 'processing', 'output']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Set defaults for optional sections
    if 'tokenization' not in config:
        config['tokenization'] = {}
    
    logger.debug(f"Configuration loaded successfully")
    
    return config


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values.
    
    Returns:
        Dictionary with default configuration
    """
    return {
        'input': {
            'raw_data_dir': 'training_materials',
            'file_patterns': ['*.rtf', '*.txt']
        },
        'processing': {
            'remove_headers': True,
            'remove_footers': True,
            'normalize_whitespace': True,
            'min_text_length': 1,  # Only used if filter_by_length=True
            'max_text_length': None,  # None = no truncation (formatting handles chunking)
            'filter_by_length': False,  # False to preserve plays/dialogue with short lines
            'chunk_size': 512,
            'chunk_overlap': 50,
            'encoding': 'utf-8',
            'handle_encoding_errors': 'replace'
        },
        'output': {
            'output_dir': 'data/processed',
            'format': 'jsonl',
            'train_ratio': 0.8,
            'validation_ratio': 0.1,
            'test_ratio': 0.1,
            'random_seed': 42
        },
        'tokenization': {
            'tokenizer_name': 'gpt2',
            'model_max_length': None,  # None = use tokenizer default
            'pre_tokenize': False,
            'cache_tokenized': True,
            'cache_dir': 'data/cache',
            'special_tokens': None  # Optional dict of special tokens
        }
    }
