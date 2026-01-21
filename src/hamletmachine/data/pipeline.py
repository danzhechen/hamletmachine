"""
Main data processing pipeline.

This module orchestrates the complete data processing pipeline:
extraction -> cleaning -> formatting -> splitting
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from .extractor import RTFExtractor
from .cleaner import TextCleaner
from .formatter import DatasetFormatter
from .splitter import DatasetSplitter
from .tokenizer_setup import TokenizerManager
from .config_loader import load_data_config, get_default_config

logger = logging.getLogger(__name__)


class DataProcessingPipeline:
    """
    Main pipeline for processing training data.
    
    This class orchestrates the complete data processing workflow:
    1. Extract text from RTF files
    2. Clean extracted text
    3. Format for Hugging Face Datasets
    4. Split into train/val/test sets
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data processing pipeline.
        
        Args:
            config: Configuration dictionary. If None, uses defaults.
        """
        if config is None:
            config = get_default_config()
        
        self.config = config
        
        # Initialize extractor
        processing_config = config.get('processing', {})
        self.extractor = RTFExtractor(
            encoding=processing_config.get('encoding', 'utf-8'),
            handle_encoding_errors=processing_config.get('handle_encoding_errors', 'replace')
        )
        
        # Initialize cleaner
        self.cleaner = TextCleaner(
            remove_headers=processing_config.get('remove_headers', True),
            remove_footers=processing_config.get('remove_footers', True),
            normalize_whitespace=processing_config.get('normalize_whitespace', True),
            min_text_length=processing_config.get('min_text_length', 1),
            max_text_length=processing_config.get('max_text_length', None),
            filter_by_length=processing_config.get('filter_by_length', False),
            handle_encoding_errors=processing_config.get('handle_encoding_errors', 'replace')
        )
        
        # Initialize tokenizer manager
        output_config = config.get('output', {})
        tokenization_config = config.get('tokenization', {})
        cache_dir = tokenization_config.get('cache_dir', 'data/cache')
        self.tokenizer_manager = TokenizerManager(
            tokenizer_name=tokenization_config.get('tokenizer_name', 'gpt2'),
            model_max_length=tokenization_config.get('model_max_length'),
            cache_dir=Path(cache_dir) if cache_dir else None,
            special_tokens=tokenization_config.get('special_tokens')
        )
        
        # Store tokenization config for later use
        self.pre_tokenize = tokenization_config.get('pre_tokenize', False)
        self.cache_tokenized = tokenization_config.get('cache_tokenized', True)
        
        # Initialize formatter
        output_dir = output_config.get('output_dir', 'data/processed')
        self.formatter = DatasetFormatter(
            chunk_size=processing_config.get('chunk_size', 512),
            chunk_overlap=processing_config.get('chunk_overlap', 50),
            tokenizer_manager=self.tokenizer_manager,
            output_format=output_config.get('format', 'jsonl'),
            output_dir=Path(output_dir) if output_dir else None
        )
        
        # Initialize splitter
        self.splitter = DatasetSplitter(
            train_ratio=output_config.get('train_ratio', 0.8),
            validation_ratio=output_config.get('validation_ratio', 0.1),
            test_ratio=output_config.get('test_ratio', 0.1),
            random_seed=output_config.get('random_seed', 42),
            shuffle=True
        )
    
    def run_extraction_and_cleaning(
        self,
        input_dir: Optional[Path] = None,
        file_patterns: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run extraction and cleaning steps.
        
        Args:
            input_dir: Directory containing RTF files. If None, uses config.
            file_patterns: File patterns to match. If None, uses config.
            
        Returns:
            List of cleaned text dictionaries with metadata
        """
        # Get configuration
        input_config = self.config.get('input', {})
        if input_dir is None:
            input_dir = Path(input_config.get('raw_data_dir', 'training_materials'))
        else:
            input_dir = Path(input_dir)
        
        if file_patterns is None:
            file_patterns = input_config.get('file_patterns', ['*.rtf'])
        
        logger.info(f"Starting data extraction and cleaning pipeline")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"File patterns: {file_patterns}")
        
        # Step 1: Extract text from RTF files
        logger.info("Step 1: Extracting text from RTF files...")
        extracted_data = self.extractor.extract_with_source_tracking(
            directory=input_dir,
            file_patterns=file_patterns
        )
        
        if not extracted_data:
            logger.warning("No text extracted from files")
            return []
        
        logger.info(f"Extracted text from {len(extracted_data)} files")
        
        # Step 2: Clean extracted text
        logger.info("Step 2: Cleaning extracted text...")
        cleaned_data = self.cleaner.clean_batch(extracted_data)
        
        if not cleaned_data:
            logger.warning("No text passed cleaning filters")
            return []
        
        logger.info(f"Cleaned {len(cleaned_data)} texts")
        
        # Log statistics
        total_chars = sum(len(item['text']) for item in cleaned_data)
        logger.info(f"Total cleaned text: {total_chars:,} characters")
        
        return cleaned_data
    
    def run_full_pipeline(
        self,
        input_dir: Optional[Path] = None,
        file_patterns: Optional[List[str]] = None,
        save_dataset: bool = True,
        split_dataset: bool = True
    ):
        """
        Run the complete pipeline: extraction -> cleaning -> formatting -> splitting -> saving.
        
        Args:
            input_dir: Directory containing RTF files. If None, uses config.
            file_patterns: File patterns to match. If None, uses config.
            save_dataset: Whether to save the formatted dataset to disk.
            split_dataset: Whether to split into train/val/test sets (default: True).
            
        Returns:
            Dictionary with:
            - 'cleaned_data': List of cleaned text dictionaries
            - 'dataset': Full Hugging Face Dataset object (if split_dataset=False)
            - 'train_dataset': Training dataset (if split_dataset=True)
            - 'validation_dataset': Validation dataset (if split_dataset=True)
            - 'test_dataset': Test dataset (if split_dataset=True)
            - 'statistics': Split statistics (if split_dataset=True)
            - 'saved_paths': Dictionary of saved file paths
        """
        # Step 1 & 2: Extract and clean
        cleaned_data = self.run_extraction_and_cleaning(
            input_dir=input_dir,
            file_patterns=file_patterns
        )
        
        if not cleaned_data:
            logger.warning("No cleaned data to format")
            return {
                'cleaned_data': cleaned_data,
                'dataset': None,
                'saved_paths': {}
            }
        
        # Step 3: Format into Hugging Face Dataset
        logger.info("Step 3: Formatting data into Hugging Face Dataset...")
        try:
            # Check cache if enabled and pre-tokenization is requested
            dataset = None
            cache_used = False
            dataset_name = "dataset"
            
            if self.cache_tokenized and self.pre_tokenize:
                cached_dataset = self.tokenizer_manager.load_cached_dataset(dataset_name)
                if cached_dataset is not None:
                    logger.info("Using cached tokenized dataset")
                    dataset = cached_dataset
                    cache_used = True
            
            # Create dataset if not cached
            if dataset is None:
                # Get tokenization parameters
                tokenization_config = self.config.get('tokenization', {})
                max_length = tokenization_config.get('model_max_length')
                
                dataset = self.formatter.create_dataset(
                    cleaned_data,
                    pre_tokenize=self.pre_tokenize,
                    max_length=max_length,
                    truncation=True,
                    padding=False  # Don't pad during preprocessing
                )
                
                # Cache if enabled and pre-tokenized
                if self.cache_tokenized and self.pre_tokenize:
                    self.tokenizer_manager.save_cached_dataset(dataset, dataset_name)
            
            result = {
                'cleaned_data': cleaned_data,
                'dataset': dataset,
                'saved_paths': {},
                'cache_used': cache_used
            }
            
            # Step 4: Split dataset
            if split_dataset:
                logger.info("Step 4: Splitting dataset into train/validation/test...")
                train_dataset, validation_dataset, test_dataset = self.splitter.split_dataset(dataset)
                
                # Compute and print statistics
                statistics = self.splitter.compute_split_statistics(
                    train_dataset, validation_dataset, test_dataset
                )
                self.splitter.print_statistics(statistics)
                
                result['train_dataset'] = train_dataset
                result['validation_dataset'] = validation_dataset
                result['test_dataset'] = test_dataset
                result['statistics'] = statistics
                
                # Step 5: Save splits (with caching if enabled)
                if save_dataset and self.formatter.output_dir:
                    logger.info("Step 5: Saving splits to disk...")
                    output_config = self.config.get('output', {})
                    
                    # Cache tokenized splits if enabled
                    if self.cache_tokenized and self.pre_tokenize:
                        self.tokenizer_manager.save_cached_dataset(train_dataset, dataset_name, "train")
                        self.tokenizer_manager.save_cached_dataset(validation_dataset, dataset_name, "validation")
                        self.tokenizer_manager.save_cached_dataset(test_dataset, dataset_name, "test")
                    
                    saved_paths = self.splitter.save_splits(
                        train_dataset=train_dataset,
                        validation_dataset=validation_dataset,
                        test_dataset=test_dataset,
                        output_dir=self.formatter.output_dir,
                        output_format=output_config.get('format', 'jsonl'),
                        formatter=self.formatter
                    )
                    result['saved_paths'] = saved_paths
                    logger.info(f"Splits saved to {self.formatter.output_dir}")
            else:
                # Save full dataset without splitting
                if save_dataset and self.formatter.output_dir:
                    logger.info("Step 4: Saving full dataset to disk...")
                    saved_path = self.formatter.save_dataset(dataset, filename="dataset")
                    result['saved_paths'] = {'dataset': saved_path}
                    logger.info(f"Dataset saved to {self.formatter.output_dir}")
            
            return result
        except Exception as e:
            logger.error(f"Error during formatting/splitting: {e}", exc_info=True)
            return {
                'cleaned_data': cleaned_data,
                'dataset': None,
                'saved_paths': {}
            }
    
    @classmethod
    def from_config_file(cls, config_path: Path) -> 'DataProcessingPipeline':
        """
        Create pipeline from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Initialized DataProcessingPipeline instance
        """
        config = load_data_config(config_path)
        return cls(config=config)
