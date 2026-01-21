"""
Dataset splitting module.

This module handles splitting datasets into train/validation/test sets
with deterministic, reproducible splits.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import random

try:
    from datasets import Dataset
except ImportError:
    raise ImportError(
        "Hugging Face Datasets library not installed. Install with: "
        "pip install datasets"
    )

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """
    Splits datasets into train/validation/test sets.
    
    This class handles:
    - Deterministic splitting with random seed
    - Configurable split ratios
    - Split statistics reporting
    - Saving splits to separate files
    """
    
    def __init__(
        self,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        shuffle: bool = True
    ):
        """
        Initialize the dataset splitter.
        
        Args:
            train_ratio: Proportion of data for training (default: 0.8)
            validation_ratio: Proportion of data for validation (default: 0.1)
            test_ratio: Proportion of data for testing (default: 0.1)
            random_seed: Random seed for reproducibility (default: 42)
            shuffle: Whether to shuffle data before splitting (default: True)
            
        Raises:
            ValueError: If ratios don't sum to 1.0
        """
        # Validate ratios
        total_ratio = train_ratio + validation_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total_ratio:.6f}. "
                f"train={train_ratio}, validation={validation_ratio}, test={test_ratio}"
            )
        
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.shuffle = shuffle
        
        logger.info(
            f"Initialized DatasetSplitter with ratios: "
            f"train={train_ratio}, validation={validation_ratio}, test={test_ratio}, "
            f"seed={random_seed}, shuffle={shuffle}"
        )
    
    def split_dataset(
        self,
        dataset: Dataset
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            dataset: Hugging Face Dataset to split
            
        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset)
        """
        if len(dataset) == 0:
            raise ValueError("Cannot split empty dataset")
        
        logger.info(f"Splitting dataset with {len(dataset)} examples...")
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        
        # Create indices
        indices = list(range(len(dataset)))
        
        # Shuffle if requested
        if self.shuffle:
            random.shuffle(indices)
            logger.debug(f"Shuffled dataset with seed {self.random_seed}")
        
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(total_size * self.train_ratio)
        validation_size = int(total_size * self.validation_ratio)
        # Test gets the remainder to account for rounding
        
        # Split indices
        train_indices = indices[:train_size]
        validation_indices = indices[train_size:train_size + validation_size]
        test_indices = indices[train_size + validation_size:]
        
        logger.info(
            f"Split sizes: train={len(train_indices)}, "
            f"validation={len(validation_indices)}, test={len(test_indices)}"
        )
        
        # Create split datasets
        train_dataset = dataset.select(train_indices)
        validation_dataset = dataset.select(validation_indices)
        test_dataset = dataset.select(test_indices)
        
        return train_dataset, validation_dataset, test_dataset
    
    def compute_split_statistics(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        test_dataset: Dataset
    ) -> Dict[str, Any]:
        """
        Compute statistics for each split.
        
        Args:
            train_dataset: Training dataset
            validation_dataset: Validation dataset
            test_dataset: Test dataset
            
        Returns:
            Dictionary with statistics for each split
        """
        def _compute_stats(dataset: Dataset, split_name: str) -> Dict[str, Any]:
            """Compute statistics for a single split."""
            if len(dataset) == 0:
                return {
                    'split': split_name,
                    'num_examples': 0,
                    'total_tokens': 0,
                    'avg_text_length': 0.0,
                    'source_files': {}
                }
            
            # Get text lengths and token counts
            text_lengths = []
            token_counts = []
            source_files = {}
            
            for example in dataset:
                text_length = example.get('text_length', len(example.get('text', '')))
                token_count = example.get('token_count', 0)
                source_file = example.get('source_file', 'unknown')
                
                text_lengths.append(text_length)
                if token_count > 0:
                    token_counts.append(token_count)
                
                source_files[source_file] = source_files.get(source_file, 0) + 1
            
            stats = {
                'split': split_name,
                'num_examples': len(dataset),
                'total_tokens': sum(token_counts) if token_counts else 0,
                'avg_text_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0.0,
                'source_files': source_files
            }
            
            if token_counts:
                stats['avg_token_count'] = sum(token_counts) / len(token_counts)
            else:
                stats['avg_token_count'] = 0.0
            
            return stats
        
        train_stats = _compute_stats(train_dataset, 'train')
        validation_stats = _compute_stats(validation_dataset, 'validation')
        test_stats = _compute_stats(test_dataset, 'test')
        
        # Compute totals
        total_examples = (
            train_stats['num_examples'] +
            validation_stats['num_examples'] +
            test_stats['num_examples']
        )
        total_tokens = (
            train_stats['total_tokens'] +
            validation_stats['total_tokens'] +
            test_stats['total_tokens']
        )
        
        statistics = {
            'train': train_stats,
            'validation': validation_stats,
            'test': test_stats,
            'total': {
                'num_examples': total_examples,
                'total_tokens': total_tokens
            }
        }
        
        return statistics
    
    def print_statistics(
        self,
        statistics: Dict[str, Any]
    ) -> None:
        """
        Print split statistics in a human-readable format.
        
        Args:
            statistics: Statistics dictionary from compute_split_statistics
        """
        logger.info("=" * 60)
        logger.info("Dataset Split Statistics")
        logger.info("=" * 60)
        
        for split_name in ['train', 'validation', 'test']:
            stats = statistics[split_name]
            logger.info(f"\n{split_name.upper()} Split:")
            logger.info(f"  Examples: {stats['num_examples']:,}")
            logger.info(f"  Total tokens: {stats['total_tokens']:,}")
            logger.info(f"  Avg text length: {stats['avg_text_length']:.1f} chars")
            logger.info(f"  Avg token count: {stats['avg_token_count']:.1f} tokens")
            
            if stats['source_files']:
                logger.info(f"  Source files:")
                for source_file, count in sorted(stats['source_files'].items()):
                    logger.info(f"    {source_file}: {count:,} examples")
        
        total = statistics['total']
        logger.info(f"\nTOTAL:")
        logger.info(f"  Examples: {total['num_examples']:,}")
        logger.info(f"  Total tokens: {total['total_tokens']:,}")
        logger.info("=" * 60)
    
    def save_splits(
        self,
        train_dataset: Dataset,
        validation_dataset: Dataset,
        test_dataset: Dataset,
        output_dir: Path,
        output_format: str = "jsonl",
        formatter: Optional[Any] = None
    ) -> Dict[str, Path]:
        """
        Save train/validation/test splits to disk.
        
        Args:
            train_dataset: Training dataset
            validation_dataset: Validation dataset
            test_dataset: Test dataset
            output_dir: Directory to save splits
            output_format: Output format ("jsonl", "parquet", or "arrow")
            formatter: Optional DatasetFormatter instance for saving
                      (if None, uses basic save methods)
            
        Returns:
            Dictionary mapping split names to saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        # Use formatter if provided, otherwise use basic save
        if formatter is not None:
            # Use formatter's save_dataset method
            saved_paths['train'] = formatter.save_dataset(
                train_dataset, filename="train"
            )
            saved_paths['validation'] = formatter.save_dataset(
                validation_dataset, filename="validation"
            )
            saved_paths['test'] = formatter.save_dataset(
                test_dataset, filename="test"
            )
        else:
            # Basic save using dataset methods
            if output_format == "jsonl":
                train_path = output_dir / "train.jsonl"
                validation_path = output_dir / "validation.jsonl"
                test_path = output_dir / "test.jsonl"
                
                train_dataset.to_json(train_path, orient="records", lines=True, index=False)
                validation_dataset.to_json(validation_path, orient="records", lines=True, index=False)
                test_dataset.to_json(test_path, orient="records", lines=True, index=False)
                
                saved_paths['train'] = train_path
                saved_paths['validation'] = validation_path
                saved_paths['test'] = test_path
                
            elif output_format == "parquet":
                train_path = output_dir / "train.parquet"
                validation_path = output_dir / "validation.parquet"
                test_path = output_dir / "test.parquet"
                
                train_dataset.to_parquet(train_path)
                validation_dataset.to_parquet(validation_path)
                test_dataset.to_parquet(test_path)
                
                saved_paths['train'] = train_path
                saved_paths['validation'] = validation_path
                saved_paths['test'] = test_path
                
            elif output_format == "arrow":
                train_path = output_dir / "train.arrow"
                validation_path = output_dir / "validation.arrow"
                test_path = output_dir / "test.arrow"
                
                train_dataset.save_to_disk(str(train_path))
                validation_dataset.save_to_disk(str(validation_path))
                test_dataset.save_to_disk(str(test_path))
                
                saved_paths['train'] = train_path
                saved_paths['validation'] = validation_path
                saved_paths['test'] = test_path
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info(f"Saved splits to {output_dir}:")
        for split_name, path in saved_paths.items():
            logger.info(f"  {split_name}: {path}")
        
        return saved_paths
