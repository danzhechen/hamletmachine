"""
Dataset loading utilities for training.

This module provides functions and classes for loading processed datasets
and preparing them for training with Hugging Face Trainer.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import json

try:
    from datasets import Dataset, load_dataset, IterableDataset
    from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer
except ImportError:
    raise ImportError(
        "Hugging Face libraries not installed. Install with: "
        "pip install datasets transformers"
    )

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads processed datasets for training.
    
    This class handles:
    - Loading datasets from various formats (JSONL/Parquet/Arrow)
    - Loading train/validation/test splits
    - Streaming datasets for large datasets
    - Format detection and transparent handling
    """
    
    def __init__(
        self,
        data_dir: Path,
        format: Optional[str] = None
    ):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir: Directory containing processed dataset files
            format: Dataset format ("jsonl", "parquet", or "arrow").
                   If None, auto-detects from files
        """
        self.data_dir = Path(data_dir)
        self.format = format
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        logger.info(f"Initialized DatasetLoader with data_dir: {self.data_dir}")
    
    def _detect_format(self, file_path: Path) -> str:
        """
        Detect dataset format from file extension.
        
        Args:
            file_path: Path to dataset file
            
        Returns:
            Format string ("jsonl", "parquet", or "arrow")
        """
        suffix = file_path.suffix.lower()
        
        if suffix == '.jsonl':
            return 'jsonl'
        elif suffix == '.parquet':
            return 'parquet'
        elif suffix in ['.arrow', '.feather']:
            return 'arrow'
        else:
            raise ValueError(f"Unknown dataset format: {suffix}")
    
    def _find_split_file(self, split_name: str) -> Optional[Path]:
        """
        Find dataset file for a given split.
        
        Args:
            split_name: Split name ("train", "validation", "test")
            
        Returns:
            Path to split file, or None if not found
        """
        # Try different format extensions
        for ext in ['.jsonl', '.parquet', '.arrow']:
            file_path = self.data_dir / f"{split_name}{ext}"
            if file_path.exists():
                return file_path
        
        return None
    
    def load_split(
        self,
        split_name: str,
        streaming: bool = False
    ) -> Dataset:
        """
        Load a dataset split.
        
        Args:
            split_name: Split name ("train", "validation", "test")
            streaming: Whether to load as streaming dataset (for large datasets)
            
        Returns:
            Hugging Face Dataset or IterableDataset
            
        Raises:
            FileNotFoundError: If split file not found
        """
        file_path = self._find_split_file(split_name)
        
        if file_path is None:
            raise FileNotFoundError(
                f"Split file not found for '{split_name}' in {self.data_dir}. "
                f"Expected: {split_name}.jsonl, {split_name}.parquet, or {split_name}.arrow"
            )
        
        format_type = self.format or self._detect_format(file_path)
        
        logger.info(f"Loading {split_name} split from {file_path} (format: {format_type}, streaming: {streaming})")
        
        try:
            if format_type == 'jsonl':
                if streaming:
                    dataset = load_dataset('json', data_files=str(file_path), split='train', streaming=True)
                else:
                    # Load JSONL file directly
                    import json
                    data = []
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                data.append(json.loads(line))
                    dataset = Dataset.from_list(data)
            elif format_type == 'parquet':
                if streaming:
                    dataset = load_dataset('parquet', data_files=str(file_path), split='train', streaming=True)
                else:
                    dataset = load_dataset('parquet', data_files=str(file_path), split='train')
            elif format_type == 'arrow':
                if streaming:
                    # Arrow format doesn't support streaming directly, load normally
                    logger.warning("Arrow format doesn't support streaming, loading normally")
                    dataset = Dataset.load_from_disk(str(file_path))
                else:
                    dataset = Dataset.load_from_disk(str(file_path))
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            logger.info(f"Loaded {split_name} split with {len(dataset) if not streaming else 'streaming'} examples")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load {split_name} split: {e}")
            raise
    
    def load_all_splits(
        self,
        streaming: bool = False
    ) -> Dict[str, Dataset]:
        """
        Load all dataset splits (train, validation, test).
        
        Args:
            streaming: Whether to load as streaming datasets
            
        Returns:
            Dictionary mapping split names to datasets
        """
        splits = {}
        
        for split_name in ['train', 'validation', 'test']:
            try:
                splits[split_name] = self.load_split(split_name, streaming=streaming)
            except FileNotFoundError as e:
                logger.warning(f"Could not load {split_name} split: {e}")
                # Continue loading other splits
        
        if not splits:
            raise FileNotFoundError(
                f"No dataset splits found in {self.data_dir}. "
                f"Expected at least one of: train.jsonl, validation.jsonl, test.jsonl"
            )
        
        logger.info(f"Loaded {len(splits)} dataset split(s)")
        return splits
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about available datasets.
        
        Returns:
            Dictionary with dataset information
        """
        info = {
            'data_dir': str(self.data_dir),
            'available_splits': [],
            'split_files': {}
        }
        
        for split_name in ['train', 'validation', 'test']:
            file_path = self._find_split_file(split_name)
            if file_path:
                info['available_splits'].append(split_name)
                info['split_files'][split_name] = {
                    'path': str(file_path),
                    'format': self._detect_format(file_path),
                    'size': file_path.stat().st_size if file_path.exists() else 0
                }
        
        return info


def load_datasets(
    data_dir: Path,
    splits: Optional[List[str]] = None,
    streaming: bool = False,
    format: Optional[str] = None
) -> Dict[str, Dataset]:
    """
    Convenience function to load dataset splits.
    
    Args:
        data_dir: Directory containing processed dataset files
        splits: List of splits to load (["train", "validation", "test"]).
               If None, loads all available splits
        streaming: Whether to load as streaming datasets
        format: Dataset format. If None, auto-detects
        
    Returns:
        Dictionary mapping split names to datasets
        
    Example:
        >>> datasets = load_datasets("data/processed")
        >>> train_dataset = datasets['train']
        >>> val_dataset = datasets['validation']
    """
    loader = DatasetLoader(data_dir, format=format)
    
    if splits is None:
        return loader.load_all_splits(streaming=streaming)
    else:
        result = {}
        for split_name in splits:
            try:
                result[split_name] = loader.load_split(split_name, streaming=streaming)
            except FileNotFoundError as e:
                logger.warning(f"Could not load {split_name}: {e}")
        return result


class LanguageModelingDataCollator:
    """
    Data collator for language modeling tasks.
    
    This class handles:
    - Padding sequences to same length
    - Truncation if needed
    - Dynamic batching
    - Masked language modeling (MLM) support (optional)
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mlm: bool = False,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: Optional[int] = None
    ):
        """
        Initialize the data collator.
        
        Args:
            tokenizer: Pre-trained tokenizer
            mlm: Whether to use masked language modeling (for BERT-style models)
            mlm_probability: Probability of masking tokens (if mlm=True)
            pad_to_multiple_of: Pad sequences to multiple of this number (for efficiency)
        """
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of
        
        # Use Hugging Face's built-in collator
        self.collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of
        )
        
        logger.info(
            f"Initialized LanguageModelingDataCollator "
            f"(mlm={mlm}, pad_to_multiple_of={pad_to_multiple_of})"
        )
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of examples.
        
        Args:
            features: List of example dictionaries
            
        Returns:
            Collated batch dictionary
        """
        return self.collator(features)
    
    def collate_batch(
        self,
        examples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Collate a batch of examples (explicit method).
        
        Args:
            examples: List of example dictionaries
            
        Returns:
            Collated batch dictionary with input_ids, attention_mask, labels
        """
        return self.collator(examples)


def prepare_dataset_for_training(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    text_column: str = 'text',
    max_length: Optional[int] = None,
    pre_tokenized: bool = False
) -> Dataset:
    """
    Prepare dataset for training (tokenize if not already tokenized).
    
    Args:
        dataset: Hugging Face Dataset
        tokenizer: Pre-trained tokenizer
        text_column: Name of column containing text (if not pre-tokenized)
        max_length: Maximum sequence length
        pre_tokenized: Whether dataset is already tokenized (has input_ids)
        
    Returns:
        Prepared dataset ready for training
    """
    if pre_tokenized:
        # Dataset is already tokenized
        if 'input_ids' not in dataset.features:
            raise ValueError(
                "Dataset marked as pre-tokenized but 'input_ids' column not found. "
                "Available columns: " + ", ".join(dataset.column_names)
            )
        logger.info("Dataset is already tokenized, skipping tokenization")
        return dataset
    
    # Tokenize the dataset
    logger.info(f"Tokenizing dataset (text_column='{text_column}', max_length={max_length})...")
    
    max_length = max_length or tokenizer.model_max_length
    
    def tokenize_function(examples):
        texts = examples[text_column]
        tokenized = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=False,  # Don't pad here, collator will handle it
            return_attention_mask=True
        )
        return tokenized
    
    # Get columns to remove (all except text_column)
    columns_to_remove = [col for col in dataset.column_names if col != text_column]
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove,
        desc="Tokenizing"
    )
    
    # Also remove text_column after tokenization
    if text_column in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.remove_columns([text_column])
    
    logger.info(f"Tokenized dataset prepared with {len(tokenized_dataset)} examples")
    logger.info(f"Features: {list(tokenized_dataset.features.keys())}")
    
    return tokenized_dataset
