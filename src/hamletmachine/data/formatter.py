"""
Dataset formatting module.

This module handles converting cleaned text into Hugging Face Dataset format,
including text chunking, dataset creation, and persistence.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

try:
    from datasets import Dataset
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError(
        "Hugging Face libraries not installed. Install with: "
        "pip install datasets transformers"
    )

from .tokenizer_setup import TokenizerManager

logger = logging.getLogger(__name__)


class DatasetFormatter:
    """
    Formats cleaned text into Hugging Face Dataset format.
    
    This class handles:
    - Text chunking (with overlap)
    - Hugging Face Dataset creation
    - Dataset persistence (JSONL/Parquet/Arrow)
    - Metadata tracking
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        tokenizer_name: str = "gpt2",
        tokenizer_manager: Optional[TokenizerManager] = None,
        output_format: str = "jsonl",
        output_dir: Optional[Path] = None
    ):
        """
        Initialize the dataset formatter.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            tokenizer_name: Name of tokenizer to use for chunking (if tokenizer_manager not provided)
            tokenizer_manager: Optional TokenizerManager instance (if provided, overrides tokenizer_name)
            output_format: Output format ("jsonl", "parquet", or "arrow")
            output_dir: Output directory for saved datasets
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.output_format = output_format.lower()
        self.output_dir = output_dir
        
        # Validate output format
        if self.output_format not in ["jsonl", "parquet", "arrow"]:
            raise ValueError(
                f"Invalid output format: {output_format}. "
                "Must be one of: jsonl, parquet, arrow"
            )
        
        # Use provided tokenizer manager or create one
        if tokenizer_manager is not None:
            self.tokenizer_manager = tokenizer_manager
            self.tokenizer = tokenizer_manager.tokenizer
            self.tokenizer_name = tokenizer_manager.tokenizer_name
        else:
            # Create a basic tokenizer manager for backward compatibility
            try:
                self.tokenizer_manager = TokenizerManager(tokenizer_name=tokenizer_name)
                self.tokenizer = self.tokenizer_manager.tokenizer
                self.tokenizer_name = tokenizer_name
            except Exception as e:
                logger.warning(
                    f"Failed to load tokenizer {tokenizer_name}: {e}. "
                    "Falling back to character-based chunking."
                )
                self.tokenizer_manager = None
                self.tokenizer = None
                self.tokenizer_name = tokenizer_name
    
    def chunk_text(
        self,
        text: str,
        source_file: str,
        chunk_id_base: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Chunk text into appropriate sizes for language modeling.
        
        Args:
            text: Text to chunk
            source_file: Source filename for metadata
            chunk_id_base: Base ID for chunk numbering
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        
        if self.tokenizer is not None:
            # Token-based chunking
            chunks = self._chunk_by_tokens(text, source_file, chunk_id_base)
        else:
            # Character-based chunking (fallback)
            chunks = self._chunk_by_characters(text, source_file, chunk_id_base)
        
        return chunks
    
    def _chunk_by_tokens(
        self,
        text: str,
        source_file: str,
        chunk_id_base: int
    ) -> List[Dict[str, Any]]:
        """Chunk text using tokenizer."""
        # Tokenize the entire text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= self.chunk_size:
            # Text fits in one chunk
            chunk_text = text
            chunk_tokens = tokens
        else:
            # Need to split into multiple chunks
            chunk_texts = []
            chunk_tokens_list = []
            
            # Try to split at sentence boundaries first
            sentences = self._split_into_sentences(text)
            
            current_chunk_tokens = []
            current_chunk_text = []
            
            for sentence in sentences:
                sentence_tokens = self.tokenizer.encode(
                    sentence, add_special_tokens=False
                )
                
                # Check if adding this sentence would exceed chunk size
                if len(current_chunk_tokens) + len(sentence_tokens) > self.chunk_size:
                    # Save current chunk if it has content
                    if current_chunk_tokens:
                        chunk_texts.append(' '.join(current_chunk_text))
                        chunk_tokens_list.append(current_chunk_tokens)
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0 and current_chunk_tokens:
                        # Take last chunk_overlap tokens from previous chunk
                        overlap_tokens = current_chunk_tokens[-self.chunk_overlap:]
                        overlap_text = self.tokenizer.decode(
                            overlap_tokens, skip_special_tokens=True
                        )
                        current_chunk_tokens = overlap_tokens
                        current_chunk_text = [overlap_text]
                    else:
                        current_chunk_tokens = []
                        current_chunk_text = []
                
                # Add sentence to current chunk
                current_chunk_tokens.extend(sentence_tokens)
                current_chunk_text.append(sentence)
            
            # Add final chunk
            if current_chunk_tokens:
                chunk_texts.append(' '.join(current_chunk_text))
                chunk_tokens_list.append(current_chunk_tokens)
            
            # If sentence-based splitting didn't work well, fall back to token-based
            if not chunk_texts or len(chunk_texts) == 1:
                chunk_texts = []
                chunk_tokens_list = []
                
                start = 0
                while start < len(tokens):
                    end = min(start + self.chunk_size, len(tokens))
                    chunk_tokens = tokens[start:end]
                    chunk_text = self.tokenizer.decode(
                        chunk_tokens, skip_special_tokens=True
                    )
                    chunk_texts.append(chunk_text)
                    chunk_tokens_list.append(chunk_tokens)
                    
                    # Move start position with overlap
                    start = end - self.chunk_overlap
                    if start >= len(tokens):
                        break
            
            chunk_text = chunk_texts
            chunk_tokens = chunk_tokens_list
        
        # Create chunk dictionaries
        chunks = []
        if isinstance(chunk_text, str):
            # Single chunk
            chunks.append({
                'text': chunk_text,
                'source_file': source_file,
                'chunk_id': chunk_id_base,
                'text_length': len(chunk_text),
                'token_count': len(chunk_tokens) if isinstance(chunk_tokens, list) else len(chunk_tokens),
                'processing_timestamp': datetime.now().isoformat()
            })
        else:
            # Multiple chunks
            for i, (ct, ct_tokens) in enumerate(zip(chunk_text, chunk_tokens)):
                chunks.append({
                    'text': ct,
                    'source_file': source_file,
                    'chunk_id': chunk_id_base + i,
                    'text_length': len(ct),
                    'token_count': len(ct_tokens),
                    'processing_timestamp': datetime.now().isoformat()
                })
        
        return chunks
    
    def _chunk_by_characters(
        self,
        text: str,
        source_file: str,
        chunk_id_base: int
    ) -> List[Dict[str, Any]]:
        """Chunk text by characters (fallback when tokenizer unavailable)."""
        # Estimate tokens as ~4 characters per token (rough approximation)
        char_chunk_size = self.chunk_size * 4
        char_overlap = self.chunk_overlap * 4
        
        chunks = []
        start = 0
        chunk_id = chunk_id_base
        
        # Try to split at sentence boundaries
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > char_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'source_file': source_file,
                    'chunk_id': chunk_id,
                    'text_length': len(chunk_text),
                    'token_count': len(chunk_text) // 4,  # Rough estimate
                    'processing_timestamp': datetime.now().isoformat()
                })
                chunk_id += 1
                
                # Start new chunk with overlap
                if char_overlap > 0:
                    overlap_text = chunk_text[-char_overlap:]
                    current_chunk = [overlap_text]
                    current_length = len(overlap_text)
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'source_file': source_file,
                'chunk_id': chunk_id,
                'text_length': len(chunk_text),
                'token_count': len(chunk_text) // 4,  # Rough estimate
                'processing_timestamp': datetime.now().isoformat()
            })
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Uses simple regex-based sentence splitting.
        """
        # Pattern to match sentence endings
        sentence_endings = r'[.!?]+(?:\s+|$)'
        sentences = re.split(sentence_endings, text)
        
        # Clean up sentences
        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            if sent:
                cleaned.append(sent)
        
        return cleaned if cleaned else [text]
    
    def create_dataset(
        self,
        cleaned_data: List[Dict[str, Any]],
        pre_tokenize: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = False
    ) -> Dataset:
        """
        Create Hugging Face Dataset from cleaned text data.
        
        Args:
            cleaned_data: List of cleaned text dictionaries from cleaner
            pre_tokenize: Whether to tokenize the dataset (add input_ids and attention_mask)
            max_length: Maximum sequence length for tokenization (None = use tokenizer default)
            truncation: Whether to truncate sequences during tokenization
            padding: Whether to pad sequences during tokenization
            
        Returns:
            Hugging Face Dataset object (with input_ids if pre_tokenize=True)
        """
        logger.info("Creating Hugging Face Dataset from cleaned text...")
        
        # Chunk all texts
        all_chunks = []
        chunk_id_counter = 0
        
        for item in cleaned_data:
            text = item.get('text', '')
            source_file = item.get('source_file', 'unknown')
            
            if not text:
                continue
            
            # Chunk this text
            chunks = self.chunk_text(
                text=text,
                source_file=source_file,
                chunk_id_base=chunk_id_counter
            )
            
            all_chunks.extend(chunks)
            chunk_id_counter += len(chunks)
        
        if not all_chunks:
            raise ValueError("No chunks created from cleaned data")
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(cleaned_data)} texts")
        
        # Create Hugging Face Dataset
        dataset = Dataset.from_list(all_chunks)
        
        # Pre-tokenize if requested
        if pre_tokenize:
            if self.tokenizer_manager is None:
                raise ValueError(
                    "Cannot pre-tokenize: tokenizer_manager is not available. "
                    "Ensure tokenizer was loaded successfully."
                )
            
            logger.info("Pre-tokenizing dataset...")
            dataset = self.tokenizer_manager.tokenize_dataset(
                dataset=dataset,
                text_column='text',
                max_length=max_length,
                truncation=truncation,
                padding=padding,
                return_attention_mask=True
            )
            logger.info("Dataset pre-tokenized with input_ids and attention_mask")
        
        logger.info(f"Dataset created with {len(dataset)} examples")
        logger.info(f"Dataset features: {list(dataset.features.keys())}")
        
        return dataset
    
    def save_dataset(
        self,
        dataset: Dataset,
        filename: Optional[str] = None,
        split_name: Optional[str] = None
    ) -> Path:
        """
        Save dataset to disk in specified format.
        
        Args:
            dataset: Hugging Face Dataset to save
            filename: Optional custom filename (without extension)
            split_name: Optional split name (train/validation/test)
            
        Returns:
            Path to saved file
        """
        if self.output_dir is None:
            raise ValueError("output_dir must be set to save dataset")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename
        if filename is None:
            if split_name:
                filename = split_name
            else:
                filename = "dataset"
        
        # Add extension based on format
        if self.output_format == "jsonl":
            filepath = self.output_dir / f"{filename}.jsonl"
            dataset.to_json(filepath, orient="records", lines=True, index=False)
        elif self.output_format == "parquet":
            filepath = self.output_dir / f"{filename}.parquet"
            dataset.to_parquet(filepath)
        elif self.output_format == "arrow":
            filepath = self.output_dir / f"{filename}.arrow"
            dataset.save_to_disk(str(filepath))
        else:
            raise ValueError(f"Unsupported format: {self.output_format}")
        
        logger.info(f"Saved dataset to {filepath} ({self.output_format} format)")
        
        return filepath
    
    def format_and_save(
        self,
        cleaned_data: List[Dict[str, Any]],
        split_name: Optional[str] = None
    ) -> Path:
        """
        Format cleaned data and save to disk.
        
        Convenience method that combines create_dataset and save_dataset.
        
        Args:
            cleaned_data: List of cleaned text dictionaries
            split_name: Optional split name for filename
            
        Returns:
            Path to saved dataset file
        """
        dataset = self.create_dataset(cleaned_data)
        return self.save_dataset(dataset, split_name=split_name)
