"""
Text file extraction module.

This module handles reading RTF and TXT files and extracting plain text content
for use in the LLM training pipeline.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import striprtf.striprtf as rtf

logger = logging.getLogger(__name__)


class RTFExtractor:
    """
    Extracts text content from RTF and TXT files.
    
    This class handles reading RTF and TXT files, extracting plain text,
    and tracking metadata about the extraction process.
    """
    
    def __init__(
        self,
        encoding: str = "utf-8",
        handle_encoding_errors: str = "replace"
    ):
        """
        Initialize the RTF extractor.
        
        Args:
            encoding: Character encoding to use when reading files
            handle_encoding_errors: How to handle encoding errors.
                Options: "strict", "replace", "ignore"
        """
        self.encoding = encoding
        self.handle_encoding_errors = handle_encoding_errors
        
    def extract_from_file(self, file_path: Path) -> Tuple[Optional[str], Dict]:
        """
        Extract text from a single RTF or TXT file.
        
        Args:
            file_path: Path to the RTF or TXT file
            
        Returns:
            Tuple of (extracted_text, metadata_dict)
            - extracted_text: Plain text content, or None if extraction failed
            - metadata_dict: Contains source_file, success, error_message, etc.
        """
        metadata = {
            "source_file": str(file_path),
            "filename": file_path.name,
            "success": False,
            "error_message": None,
            "file_size": None,
            "text_length": None,
        }
        
        try:
            # Check if file exists
            if not file_path.exists():
                error_msg = f"File not found: {file_path}"
                logger.error(error_msg)
                metadata["error_message"] = error_msg
                return None, metadata
            
            # Get file size
            file_size = file_path.stat().st_size
            metadata["file_size"] = file_size
            
            # Determine file type and extract accordingly
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.txt':
                # Read plain text file directly
                logger.debug(f"Reading TXT file: {file_path}")
                with open(file_path, "r", encoding=self.encoding, errors=self.handle_encoding_errors) as f:
                    plain_text = f.read()
            elif file_ext == '.rtf':
                # Read RTF file and extract text
                logger.debug(f"Reading RTF file: {file_path}")
                with open(file_path, "r", encoding=self.encoding, errors=self.handle_encoding_errors) as f:
                    rtf_content = f.read()
                
                # Extract plain text from RTF
                logger.debug(f"Extracting text from RTF: {file_path}")
                plain_text = rtf.rtf_to_text(rtf_content)
            else:
                error_msg = f"Unsupported file type: {file_ext}. Supported: .rtf, .txt"
                logger.error(error_msg)
                metadata["error_message"] = error_msg
                return None, metadata
            
            # Update metadata
            metadata["success"] = True
            metadata["text_length"] = len(plain_text)
            
            logger.info(
                f"Successfully extracted {len(plain_text)} characters from {file_path.name}"
            )
            
            return plain_text, metadata
            
        except UnicodeDecodeError as e:
            error_msg = f"Encoding error reading {file_path}: {e}"
            logger.error(error_msg)
            metadata["error_message"] = error_msg
            return None, metadata
            
        except Exception as e:
            error_msg = f"Error extracting text from {file_path}: {e}"
            logger.error(error_msg, exc_info=True)
            metadata["error_message"] = str(e)
            return None, metadata
    
    def extract_from_directory(
        self,
        directory: Path,
        file_patterns: List[str] = None
    ) -> List[Tuple[Optional[str], Dict]]:
        """
        Extract text from all matching files in a directory.
        
        Args:
            directory: Directory containing RTF files
            file_patterns: List of glob patterns to match (e.g., ["*.rtf", "*.txt"])
                If None, defaults to ["*.rtf"]
                
        Returns:
            List of tuples (extracted_text, metadata_dict) for each file
        """
        if file_patterns is None:
            file_patterns = ["*.rtf"]
        
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory}")
            return []
        
        results = []
        files_processed = 0
        files_successful = 0
        
        # Find all matching files
        all_files = []
        for pattern in file_patterns:
            matching_files = list(directory.glob(pattern))
            all_files.extend(matching_files)
        
        # Remove duplicates (in case patterns overlap)
        all_files = list(set(all_files))
        all_files.sort()  # Process in consistent order
        
        logger.info(f"Found {len(all_files)} files matching patterns {file_patterns}")
        
        # Extract text from each file
        for file_path in all_files:
            files_processed += 1
            text, metadata = self.extract_from_file(file_path)
            results.append((text, metadata))
            
            if metadata["success"]:
                files_successful += 1
            else:
                logger.warning(
                    f"Failed to extract from {file_path.name}: {metadata['error_message']}"
                )
        
        logger.info(
            f"Extraction complete: {files_successful}/{files_processed} files successful"
        )
        
        return results
    
    def extract_with_source_tracking(
        self,
        directory: Path,
        file_patterns: List[str] = None
    ) -> List[Dict]:
        """
        Extract text and return structured data with source tracking.
        
        This method returns a list of dictionaries, each containing:
        - text: Extracted text content
        - source_file: Original filename
        - metadata: Additional extraction metadata
        
        Args:
            directory: Directory containing RTF or TXT files
            file_patterns: List of glob patterns to match
            
        Returns:
            List of dictionaries with text, source_file, and metadata
        """
        results = self.extract_from_directory(directory, file_patterns)
        
        structured_results = []
        for text, metadata in results:
            if text is not None:  # Only include successful extractions
                structured_results.append({
                    "text": text,
                    "source_file": metadata["filename"],
                    "metadata": metadata
                })
        
        return structured_results
