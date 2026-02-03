"""
Text cleaning module.

This module handles cleaning extracted text by removing artifacts,
normalizing whitespace, and preparing text for language modeling.
"""

import logging
import re
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Cleans text content for LLM training.
    
    This class handles various text cleaning operations including:
    - Header/footer removal
    - Whitespace normalization
    - Special character handling
    - Text length filtering
    """
    
    def __init__(
        self,
        remove_headers: bool = True,
        remove_footers: bool = True,
        normalize_whitespace: bool = True,
        min_text_length: int = 1,
        max_text_length: Optional[int] = None,
        filter_by_length: bool = False,
        handle_encoding_errors: str = "replace"
    ):
        """
        Initialize the text cleaner.
        
        Args:
            remove_headers: Whether to attempt header removal
            remove_footers: Whether to attempt footer removal
            normalize_whitespace: Whether to normalize whitespace
            min_text_length: Minimum character length for filtering (only used if filter_by_length=True)
            max_text_length: Maximum character length for truncation (None = no truncation)
            filter_by_length: Whether to filter out short texts (False by default to preserve plays/dialogue)
            handle_encoding_errors: How to handle encoding errors
        """
        self.remove_headers = remove_headers
        self.remove_footers = remove_footers
        self.normalize_whitespace = normalize_whitespace
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.filter_by_length = filter_by_length
        self.handle_encoding_errors = handle_encoding_errors
        
    def clean(self, text: str, source_file: Optional[str] = None) -> Tuple[Optional[str], Dict]:
        """
        Clean a single text string.
        
        Args:
            text: Raw text to clean
            source_file: Optional source filename for logging
            
        Returns:
            Tuple of (cleaned_text, statistics_dict)
            - cleaned_text: Cleaned text, or None if filtered out
            - statistics_dict: Cleaning statistics
        """
        if not text:
            return None, {"original_length": 0, "filtered": True, "reason": "empty"}
        
        original_length = len(text)
        stats = {
            "original_length": original_length,
            "filtered": False,
            "reason": None,
            "operations_applied": []
        }
        
        cleaned = text

        # Project Gutenberg: strip everything before "*** START OF ... ***" and after "*** END OF ... ***"
        cleaned = self._strip_project_gutenberg_boilerplate(cleaned)
        if len(cleaned) < original_length:
            stats["operations_applied"].append("pg_boilerplate_removal")
        
        # Remove headers (first N lines, typically copyright/header info)
        if self.remove_headers:
            cleaned = self._remove_headers(cleaned)
            if len(cleaned) < original_length:
                stats["operations_applied"].append("header_removal")
        
        # Remove footers (last N lines, typically page numbers/footer info)
        if self.remove_footers:
            cleaned = self._remove_footers(cleaned)
            if len(cleaned) < len(text):
                stats["operations_applied"].append("footer_removal")
        
        # Normalize whitespace
        if self.normalize_whitespace:
            cleaned = self._normalize_whitespace(cleaned)
            stats["operations_applied"].append("whitespace_normalization")
        
        # Handle encoding errors
        cleaned = self._handle_encoding_errors(cleaned)
        
        # Filter by length (only if explicitly enabled - disabled by default to preserve plays/dialogue)
        if self.filter_by_length and len(cleaned) < self.min_text_length:
            stats["filtered"] = True
            stats["reason"] = f"too_short ({len(cleaned)} < {self.min_text_length})"
            logger.debug(
                f"Filtered text from {source_file}: {stats['reason']}"
            )
            return None, stats
        
        # Truncate if max length is set (usually not needed as formatting handles chunking)
        if self.max_text_length is not None and len(cleaned) > self.max_text_length:
            cleaned = cleaned[:self.max_text_length]
            stats["operations_applied"].append("truncation")
            logger.debug(
                f"Truncated text from {source_file}: {len(cleaned)} > {self.max_text_length}"
            )
        
        stats["final_length"] = len(cleaned)
        stats["characters_removed"] = original_length - len(cleaned)
        
        return cleaned, stats

    def _strip_project_gutenberg_boilerplate(self, text: str) -> str:
        """
        Remove Project Gutenberg header/footer: keep only content between
        "*** START OF THE PROJECT GUTENBERG EBOOK ... ***" and
        "*** END OF THE PROJECT GUTENBERG EBOOK ... ***".
        If markers are not found, return text unchanged.
        """
        lines = text.split('\n')
        start_i = None
        end_i = None
        for i, line in enumerate(lines):
            stripped = line.strip().upper()
            if '*** START OF' in stripped and 'PROJECT GUTENBERG' in stripped:
                start_i = i + 1  # content starts after the START line
                break
        for i in range(len(lines) - 1, -1, -1):
            stripped = lines[i].strip().upper()
            if '*** END OF' in stripped and 'PROJECT GUTENBERG' in stripped:
                end_i = i  # content ends before the END line
                break
        if start_i is not None and end_i is not None and start_i < end_i:
            result = '\n'.join(lines[start_i:end_i])
            logger.debug(
                "Removed Project Gutenberg boilerplate: kept lines %s-%s (of %s)",
                start_i + 1, end_i, len(lines)
            )
            return result
        return text
    
    def _remove_headers(self, text: str, lines_to_check: int = 15) -> str:
        """
        Attempt to remove headers from the beginning of text.
        
        Headers are typically:
        - URLs, copyright notices, or metadata
        - Project Gutenberg headers
        - License information
        - Table of contents indicators
        - Appear at the very beginning
        
        Args:
            text: Text to process
            lines_to_check: Number of initial lines to check for header patterns
            
        Returns:
            Text with headers removed
        """
        lines = text.split('\n')
        if len(lines) <= lines_to_check:
            return text  # Too short to have headers
        
        # Common header patterns (more comprehensive)
        header_patterns = [
            r'^https?://',  # URLs
            r'^www\.',  # URLs without protocol
            r'^Copyright',  # Copyright notices
            r'^\(C\)',  # Copyright symbol
            r'^\d{1,2}$',  # Page numbers (single or double digits)
            r'^Page \d+',  # "Page X"
            r'^Page \d+ of \d+',  # "Page X of Y"
            r'^Project Gutenberg',  # Project Gutenberg headers
            r'^Etext',  # Etext references
            r'^Release Date',  # Release date lines
            r'^\[.*?\]$',  # Bracketed metadata like [Title], [Author]
            r'^Title:',  # Title metadata
            r'^Author:',  # Author metadata
            r'^Language:',  # Language metadata
            r'^Character set encoding:',  # Encoding metadata
            r'^\s*$',  # Empty lines
            r'^[*=_-]{3,}',  # Separator lines (***, ===, ___, ---)
        ]
        
        # Check first few lines for header patterns
        header_end = 0
        consecutive_header_lines = 0
        max_consecutive_headers = 3  # Allow some header lines but stop after content starts
        
        for i, line in enumerate(lines[:lines_to_check]):
            line_stripped = line.strip()
            
            # Skip empty lines (but don't count as breaking the header)
            if not line_stripped:
                if i < 5:  # Empty lines at the very beginning are likely headers
                    header_end = i + 1
                continue
            
            # Check if line matches header patterns
            is_header = False
            for pattern in header_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    is_header = True
                    break
            
            # Additional heuristics for headers
            # - Lines that are mostly uppercase and short (likely titles/headers)
            if len(line_stripped) < 80 and line_stripped.isupper() and len(line_stripped.split()) < 10:
                is_header = True
            
            # - Lines containing email addresses
            if '@' in line_stripped and ('http' in line_stripped.lower() or 'www' in line_stripped.lower()):
                is_header = True
            
            if is_header:
                header_end = i + 1
                consecutive_header_lines += 1
            else:
                # Found non-header content
                # If we've seen several header lines and now see content, we're done
                if consecutive_header_lines >= 2:
                    break
                # If this looks like actual content (not a header), stop
                if len(line_stripped) > 20 or any(c.islower() for c in line_stripped):
                    # Has lowercase letters or is substantial - likely content
                    break
                # Otherwise, might still be in header area, continue checking
        
        if header_end > 0:
            logger.debug(f"Removed {header_end} header lines")
            return '\n'.join(lines[header_end:])
        
        return text
    
    def _remove_footers(self, text: str, lines_to_check: int = 15) -> str:
        """
        Attempt to remove footers from the end of text.
        
        Footers are typically:
        - Page numbers
        - Copyright notices
        - Project Gutenberg footers
        - End-of-file markers
        
        Args:
            text: Text to process
            lines_to_check: Number of final lines to check for footer patterns
            
        Returns:
            Text with footers removed
        """
        lines = text.split('\n')
        if len(lines) <= lines_to_check:
            return text  # Too short to have footers
        
        # Common footer patterns (more comprehensive)
        footer_patterns = [
            r'^\d+$',  # Page numbers (standalone)
            r'^Page \d+',  # "Page X"
            r'^Page \d+ of \d+',  # "Page X of Y"
            r'^-\s*\d+\s*-',  # "- X -" page numbers
            r'^\s*$',  # Empty lines
            r'^Copyright',  # Copyright notices
            r'^\(C\)',  # Copyright symbol
            r'^End of.*Project Gutenberg',  # End markers
            r'^End of the Project Gutenberg',  # End markers
            r'^\*\*\* END OF.*\*\*\*',  # End markers
            r'^\[.*?\]$',  # Bracketed metadata
            r'^[*=_-]{3,}',  # Separator lines
        ]
        
        # Check last few lines for footer patterns
        footer_start = len(lines)
        consecutive_footer_lines = 0
        
        for i in range(len(lines) - 1, max(-1, len(lines) - lines_to_check - 1), -1):
            line_stripped = lines[i].strip()
            
            # Skip empty lines at the end
            if not line_stripped:
                if i >= len(lines) - 3:  # Empty lines at the very end are likely footers
                    footer_start = i
                continue
            
            # Check if line matches footer patterns
            is_footer = False
            for pattern in footer_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    is_footer = True
                    break
            
            # Additional heuristics for footers
            # - Very short lines at the end (but be careful with plays - short dialogue is valid)
            # Only consider it a footer if it's a number or matches patterns
            if len(line_stripped) < 20 and i >= len(lines) - 3:
                # Check if it's just a number (page number)
                if re.match(r'^\d+$', line_stripped):
                    is_footer = True
                # Check if it's all caps and short (likely footer metadata)
                elif line_stripped.isupper() and len(line_stripped.split()) <= 3:
                    is_footer = True
            
            if is_footer:
                footer_start = i
                consecutive_footer_lines += 1
            else:
                # Found non-footer content
                # If we've seen footer patterns and now see content, we're done
                if consecutive_footer_lines >= 1:
                    break
                # If this looks like actual content, stop
                if len(line_stripped) > 30 or any(c.islower() for c in line_stripped):
                    # Has lowercase letters or is substantial - likely content
                    break
        
        if footer_start < len(lines):
            removed_count = len(lines) - footer_start
            logger.debug(f"Removed {removed_count} footer lines")
            return '\n'.join(lines[:footer_start])
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.
        
        Operations:
        - Replace multiple spaces with single space
        - Normalize line breaks (handle Windows/Unix/Mac)
        - Remove excessive blank lines (max 2 consecutive)
        - Preserve intentional paragraph breaks
        
        Args:
            text: Text to normalize
            
        Returns:
            Text with normalized whitespace
        """
        # Normalize line endings to \n
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Replace multiple spaces with single space (but preserve line breaks)
        # First, replace spaces within lines
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            # Replace multiple spaces with single space
            line = re.sub(r' +', ' ', line)
            normalized_lines.append(line)
        text = '\n'.join(normalized_lines)
        
        # Remove excessive blank lines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _handle_encoding_errors(self, text: str) -> str:
        """
        Handle encoding errors in text.
        
        Args:
            text: Text to process
            
        Returns:
            Text with encoding errors handled
        """
        if self.handle_encoding_errors == "replace":
            # Replace problematic characters
            text = text.encode('utf-8', errors='replace').decode('utf-8')
        elif self.handle_encoding_errors == "ignore":
            # Ignore problematic characters
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
        # "strict" mode: let errors propagate (default behavior)
        
        return text
    
    def clean_batch(
        self,
        texts: List[Dict],
        source_file: Optional[str] = None
    ) -> List[Dict]:
        """
        Clean a batch of texts.
        
        Args:
            texts: List of dictionaries with 'text' and 'source_file' keys
            source_file: Optional source filename for logging
            
        Returns:
            List of cleaned texts with metadata
        """
        cleaned_results = []
        total_original = 0
        total_filtered = 0
        
        for item in texts:
            text = item.get("text", "")
            source = item.get("source_file", source_file)
            
            if not text:
                continue
            
            total_original += 1
            cleaned_text, stats = self.clean(text, source)
            
            if cleaned_text is not None:
                cleaned_results.append({
                    "text": cleaned_text,
                    "source_file": source,
                    "metadata": {
                        **item.get("metadata", {}),
                        "cleaning_stats": stats
                    }
                })
            else:
                total_filtered += 1
                logger.debug(f"Filtered text from {source}: {stats.get('reason')}")
        
        logger.info(
            f"Cleaned batch: {len(cleaned_results)}/{total_original} texts passed "
            f"({total_filtered} filtered)"
        )
        
        return cleaned_results
