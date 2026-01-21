"""
Tokenizer setup and management module.

This module handles tokenizer loading, configuration, testing, and caching
for the data processing pipeline.
"""

import logging
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path as PathType

try:
    from transformers import AutoTokenizer, PreTrainedTokenizer
    from datasets import Dataset
except ImportError:
    raise ImportError(
        "Hugging Face libraries not installed. Install with: "
        "pip install transformers datasets"
    )

logger = logging.getLogger(__name__)


class TokenizerManager:
    """
    Manages tokenizer loading, configuration, testing, and caching.
    
    This class handles:
    - Loading tokenizers from Hugging Face Hub or local paths
    - Configuring tokenizer parameters (model_max_length, special tokens)
    - Testing tokenizers on sample text
    - Caching tokenized datasets
    - Cache invalidation based on tokenizer changes
    """
    
    def __init__(
        self,
        tokenizer_name: str,
        model_max_length: Optional[int] = None,
        use_fast: bool = True,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        cache_dir: Optional[PathType] = None,
        special_tokens: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the tokenizer manager.
        
        Args:
            tokenizer_name: Name or path of tokenizer (HF Hub name or local path)
            model_max_length: Maximum sequence length (None = use tokenizer default)
            use_fast: Whether to use fast tokenizer (if available)
            trust_remote_code: Whether to trust remote code in tokenizer
            local_files_only: Whether to only use local files
            cache_dir: Directory for caching tokenized data
            special_tokens: Dictionary of special tokens to add (e.g., {'pad_token': '<pad>'})
        """
        self.tokenizer_name = tokenizer_name
        self.model_max_length = model_max_length
        self.use_fast = use_fast
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.special_tokens = special_tokens or {}
        
        # Load and configure tokenizer
        self.tokenizer = self._load_tokenizer()
        self._configure_tokenizer()
        
        # Compute tokenizer hash for cache invalidation
        self.tokenizer_hash = self._compute_tokenizer_hash()
        
        logger.info(f"Initialized TokenizerManager with tokenizer: {tokenizer_name}")
        logger.info(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")
        logger.info(f"Model max length: {self.model_max_length or self.tokenizer.model_max_length}")
    
    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load tokenizer from Hugging Face Hub or local path."""
        logger.info(f"Loading tokenizer: {self.tokenizer_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                use_fast=self.use_fast,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only
            )
            logger.info(f"Successfully loaded tokenizer: {self.tokenizer_name}")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer {self.tokenizer_name}: {e}")
            raise
    
    def _configure_tokenizer(self) -> None:
        """Configure tokenizer parameters."""
        # Set model_max_length
        if self.model_max_length is not None:
            self.tokenizer.model_max_length = self.model_max_length
            logger.debug(f"Set model_max_length to {self.model_max_length}")
        
        # Add special tokens
        if self.special_tokens:
            # Check if tokenizer needs to resize embeddings
            tokens_to_add = []
            for token_type, token_value in self.special_tokens.items():
                if hasattr(self.tokenizer, token_type):
                    current_token = getattr(self.tokenizer, token_type)
                    if current_token is None or current_token != token_value:
                        tokens_to_add.append(token_value)
                        setattr(self.tokenizer, token_type, token_value)
                        logger.debug(f"Set {token_type} to {token_value}")
            
            # Resize token embeddings if needed (for models that need it)
            if tokens_to_add:
                try:
                    # This is typically handled by the model, but we log it
                    logger.debug(f"Special tokens added: {tokens_to_add}")
                except Exception as e:
                    logger.warning(f"Could not resize embeddings for special tokens: {e}")
        
        # Ensure pad_token is set (common requirement)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.debug("Set pad_token to eos_token")
            else:
                logger.warning("No pad_token or eos_token available")
    
    def _compute_tokenizer_hash(self) -> str:
        """
        Compute hash of tokenizer configuration for cache invalidation.
        
        Returns:
            Hexadecimal hash string
        """
        # Create a dictionary of tokenizer properties that affect tokenization
        tokenizer_info = {
            'name': self.tokenizer_name,
            'model_max_length': self.model_max_length or self.tokenizer.model_max_length,
            'vocab_size': self.tokenizer.vocab_size,
            'special_tokens': self.special_tokens,
            'pad_token': self.tokenizer.pad_token,
            'eos_token': self.tokenizer.eos_token,
            'bos_token': self.tokenizer.bos_token,
            'unk_token': self.tokenizer.unk_token,
        }
        
        # Convert to JSON string and hash
        info_str = json.dumps(tokenizer_info, sort_keys=True)
        hash_obj = hashlib.sha256(info_str.encode('utf-8'))
        return hash_obj.hexdigest()[:16]  # Use first 16 chars for readability
    
    def test_tokenizer(
        self,
        sample_texts: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Test tokenizer on sample text.
        
        Args:
            sample_texts: List of sample texts to test. If None, uses default samples.
            verbose: Whether to print detailed results
            
        Returns:
            Dictionary with test results
        """
        if sample_texts is None:
            sample_texts = [
                "Hello, world!",
                "This is a longer sentence with multiple words.",
                "Special characters: !@#$%^&*()",
                "Numbers: 1234567890",
                "Unicode: 你好世界 🌍"
            ]
        
        results = {
            'tokenizer_name': self.tokenizer_name,
            'vocab_size': self.tokenizer.vocab_size,
            'model_max_length': self.model_max_length or self.tokenizer.model_max_length,
            'special_tokens': {
                'pad_token': self.tokenizer.pad_token,
                'eos_token': self.tokenizer.eos_token,
                'bos_token': self.tokenizer.bos_token,
                'unk_token': self.tokenizer.unk_token,
            },
            'samples': []
        }
        
        for i, text in enumerate(sample_texts):
            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            tokens_with_special = self.tokenizer.encode(text, add_special_tokens=True)
            
            # Decode
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
            
            sample_result = {
                'text': text,
                'token_count': len(tokens),
                'token_count_with_special': len(tokens_with_special),
                'tokens': tokens[:10],  # First 10 tokens
                'decoded': decoded,
                'round_trip_match': (text.strip() == decoded.strip())
            }
            results['samples'].append(sample_result)
            
            if verbose:
                logger.info(f"Sample {i+1}:")
                logger.info(f"  Text: {text[:50]}...")
                logger.info(f"  Tokens: {len(tokens)}")
                logger.info(f"  Round-trip match: {sample_result['round_trip_match']}")
        
        if verbose:
            logger.info(f"\nTokenizer Test Summary:")
            logger.info(f"  Vocab size: {results['vocab_size']}")
            logger.info(f"  Model max length: {results['model_max_length']}")
            logger.info(f"  Special tokens: {results['special_tokens']}")
        
        return results
    
    def tokenize_dataset(
        self,
        dataset: Dataset,
        text_column: str = 'text',
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = False,
        padding_style: str = 'max_length',  # 'max_length' or 'longest'
        return_attention_mask: bool = True
    ) -> Dataset:
        """
        Tokenize a Hugging Face Dataset.
        
        Args:
            dataset: Dataset to tokenize
            text_column: Name of column containing text
            max_length: Maximum sequence length (None = use tokenizer default)
            truncation: Whether to truncate sequences
            padding: Whether to pad sequences (False = no padding, True = pad to max_length)
            return_attention_mask: Whether to return attention masks
            
        Returns:
            Tokenized dataset with 'input_ids' and optionally 'attention_mask'
        """
        if text_column not in dataset.column_names:
            raise ValueError(f"Column '{text_column}' not found in dataset. Available: {dataset.column_names}")
        
        max_length = max_length or self.model_max_length or self.tokenizer.model_max_length
        
        logger.info(f"Tokenizing dataset with {len(dataset)} examples...")
        logger.info(f"  max_length: {max_length}")
        logger.info(f"  truncation: {truncation}")
        logger.info(f"  padding: {padding}")
        
        # Convert padding bool to proper padding style
        # If padding=True and max_length is set, use 'max_length' to pad to that length
        # Otherwise, use True to pad to longest sequence in batch
        if padding:
            if max_length is not None and padding_style == 'max_length':
                padding_value = 'max_length'
            else:
                padding_value = True
        else:
            padding_value = False
        
        def tokenize_function(examples):
            texts = examples[text_column]
            tokenized = self.tokenizer(
                texts,
                max_length=max_length,
                truncation=truncation,
                padding=padding_value,
                return_attention_mask=return_attention_mask
            )
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[text_column] if text_column in dataset.column_names else [],
            desc="Tokenizing"
        )
        
        logger.info(f"Tokenized dataset created with {len(tokenized_dataset)} examples")
        logger.info(f"  Features: {list(tokenized_dataset.features.keys())}")
        
        return tokenized_dataset
    
    def get_cache_path(
        self,
        dataset_name: str,
        split_name: Optional[str] = None
    ) -> Optional[PathType]:
        """
        Get cache path for tokenized dataset.
        
        Args:
            dataset_name: Name of the dataset
            split_name: Optional split name (train/validation/test)
            
        Returns:
            Path to cache file, or None if caching disabled
        """
        if self.cache_dir is None:
            return None
        
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Include tokenizer hash in filename for cache invalidation
        filename = f"{dataset_name}_{self.tokenizer_hash}"
        if split_name:
            filename = f"{split_name}_{filename}"
        filename = f"{filename}.arrow"
        
        return cache_dir / filename
    
    def load_cached_dataset(
        self,
        dataset_name: str,
        split_name: Optional[str] = None
    ) -> Optional[Dataset]:
        """
        Load cached tokenized dataset if available and valid.
        
        Args:
            dataset_name: Name of the dataset
            split_name: Optional split name
            
        Returns:
            Cached dataset if available and valid, None otherwise
        """
        cache_path = self.get_cache_path(dataset_name, split_name)
        if cache_path is None or not cache_path.exists():
            return None
        
        try:
            logger.info(f"Loading cached dataset from {cache_path}")
            dataset = Dataset.load_from_disk(str(cache_path))
            logger.info(f"Successfully loaded cached dataset with {len(dataset)} examples")
            return dataset
        except Exception as e:
            logger.warning(f"Failed to load cached dataset: {e}")
            return None
    
    def save_cached_dataset(
        self,
        dataset: Dataset,
        dataset_name: str,
        split_name: Optional[str] = None
    ) -> Optional[PathType]:
        """
        Save tokenized dataset to cache.
        
        Args:
            dataset: Dataset to cache
            dataset_name: Name of the dataset
            split_name: Optional split name
            
        Returns:
            Path to saved cache file, or None if caching disabled
        """
        cache_path = self.get_cache_path(dataset_name, split_name)
        if cache_path is None:
            return None
        
        try:
            logger.info(f"Saving dataset to cache: {cache_path}")
            dataset.save_to_disk(str(cache_path))
            logger.info(f"Successfully cached dataset with {len(dataset)} examples")
            return cache_path
        except Exception as e:
            logger.warning(f"Failed to save cached dataset: {e}")
            return None
    
    def invalidate_cache(
        self,
        dataset_name: Optional[str] = None
    ) -> int:
        """
        Invalidate cache files.
        
        Args:
            dataset_name: Optional dataset name to invalidate. If None, invalidates all.
            
        Returns:
            Number of cache files removed
        """
        if self.cache_dir is None or not self.cache_dir.exists():
            return 0
        
        cache_dir = Path(self.cache_dir)
        removed_count = 0
        
        if dataset_name:
            # Remove specific dataset cache
            pattern = f"*{dataset_name}*"
            for cache_file in cache_dir.glob(pattern):
                try:
                    cache_file.unlink()
                    removed_count += 1
                    logger.debug(f"Removed cache file: {cache_file}")
                except (PermissionError, OSError) as e:
                    logger.warning(f"Could not remove cache file {cache_file}: {e}")
        else:
            # Remove all cache files
            for cache_file in cache_dir.glob("*.arrow"):
                try:
                    cache_file.unlink()
                    removed_count += 1
                    logger.debug(f"Removed cache file: {cache_file}")
                except (PermissionError, OSError) as e:
                    logger.warning(f"Could not remove cache file {cache_file}: {e}")
        
        if removed_count > 0:
            logger.info(f"Invalidated {removed_count} cache file(s)")
        
        return removed_count
    
    def verify_compatibility(
        self,
        model_name: Optional[str] = None,
        expected_vocab_size: Optional[int] = None
    ) -> Tuple[bool, List[str]]:
        """
        Verify tokenizer compatibility with model architecture.
        
        Args:
            model_name: Optional model name to check compatibility
            expected_vocab_size: Expected vocabulary size
            
        Returns:
            Tuple of (is_compatible, list_of_warnings)
        """
        warnings = []
        is_compatible = True
        
        # Check vocab size
        if expected_vocab_size is not None:
            if self.tokenizer.vocab_size != expected_vocab_size:
                warnings.append(
                    f"Vocabulary size mismatch: tokenizer has {self.tokenizer.vocab_size}, "
                    f"expected {expected_vocab_size}"
                )
                is_compatible = False
        
        # Check for required special tokens
        if self.tokenizer.pad_token is None:
            warnings.append("No pad_token set - may cause issues during training")
            # Not necessarily incompatible, but worth warning
        
        # Check model_max_length
        max_len = self.model_max_length or self.tokenizer.model_max_length
        if max_len is None or max_len <= 0:
            warnings.append("model_max_length is not set or invalid")
            is_compatible = False
        
        if model_name:
            # Try to load model tokenizer for comparison
            try:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(model_name)
                # This is a basic check - more sophisticated checks could be added
                logger.debug(f"Model {model_name} loaded for compatibility check")
            except Exception as e:
                warnings.append(f"Could not verify compatibility with model {model_name}: {e}")
        
        if is_compatible and not warnings:
            logger.info("Tokenizer compatibility check passed")
        else:
            logger.warning(f"Tokenizer compatibility check found {len(warnings)} issue(s)")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        
        return is_compatible, warnings
