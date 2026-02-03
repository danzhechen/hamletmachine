"""
Tests for data cleaning module.
"""

import pytest
from hamletmachine.data import TextCleaner


def test_cleaner_initialization():
    """Test TextCleaner initialization."""
    cleaner = TextCleaner()
    assert cleaner.remove_headers is True
    assert cleaner.remove_footers is True
    assert cleaner.normalize_whitespace is True
    assert cleaner.min_text_length == 50
    assert cleaner.max_text_length == 10000
    
    cleaner = TextCleaner(
        remove_headers=False,
        remove_footers=False,
        min_text_length=100
    )
    assert cleaner.remove_headers is False
    assert cleaner.remove_footers is False
    assert cleaner.min_text_length == 100


def test_clean_basic_text():
    """Test cleaning basic text."""
    cleaner = TextCleaner(min_text_length=10)
    text = "This is a test text that should be cleaned properly."
    cleaned, stats = cleaner.clean(text)
    
    assert cleaned is not None
    assert len(cleaned) > 0
    assert stats["original_length"] == len(text)
    assert stats["filtered"] is False


def test_clean_too_short():
    """Test filtering text that's too short."""
    cleaner = TextCleaner(min_text_length=100)
    text = "Short text"
    cleaned, stats = cleaner.clean(text)
    
    assert cleaned is None
    assert stats["filtered"] is True
    assert "too_short" in stats["reason"]


def test_clean_empty_text():
    """Test cleaning empty text."""
    cleaner = TextCleaner()
    cleaned, stats = cleaner.clean("")
    
    assert cleaned is None
    assert stats["filtered"] is True
    assert stats["original_length"] == 0


def test_normalize_whitespace():
    """Test whitespace normalization."""
    cleaner = TextCleaner(normalize_whitespace=True, min_text_length=10)
    text = "This   has    multiple    spaces\n\n\nand   line   breaks."
    cleaned, stats = cleaner.clean(text)
    
    assert cleaned is not None
    # Should have normalized spaces
    assert "   " not in cleaned  # No triple spaces
    assert "\n\n\n" not in cleaned  # No triple line breaks
    assert "normalize_whitespace" in stats["operations_applied"]


def test_remove_headers():
    """Test header removal."""
    cleaner = TextCleaner(remove_headers=True, min_text_length=10)
    text = "https://example.com\nCopyright 2024\n\nThis is the actual content that should remain."
    cleaned, stats = cleaner.clean(text)
    
    assert cleaned is not None
    assert "https://example.com" not in cleaned
    assert "Copyright 2024" not in cleaned
    assert "actual content" in cleaned
    assert "header_removal" in stats["operations_applied"]


def test_remove_footers():
    """Test footer removal."""
    cleaner = TextCleaner(remove_footers=True, min_text_length=10)
    text = "This is the actual content that should remain.\n\nPage 1\nCopyright 2024"
    cleaned, stats = cleaner.clean(text)
    
    assert cleaned is not None
    assert "Page 1" not in cleaned
    assert "Copyright 2024" not in cleaned
    assert "actual content" in cleaned
    assert "footer_removal" in stats["operations_applied"]


def test_truncate_long_text():
    """Test truncation of very long text."""
    cleaner = TextCleaner(max_text_length=100, min_text_length=10)
    text = "A" * 200  # 200 characters
    cleaned, stats = cleaner.clean(text)
    
    assert cleaned is not None
    assert len(cleaned) == 100  # Should be truncated
    assert "truncation" in stats["operations_applied"]


def test_clean_batch():
    """Test batch cleaning."""
    cleaner = TextCleaner(min_text_length=10)
    texts = [
        {"text": "This is a valid text that should pass.", "source_file": "test1.txt"},
        {"text": "Short", "source_file": "test2.txt"},  # Too short
        {"text": "Another valid text that should also pass.", "source_file": "test3.txt"},
    ]
    
    cleaned = cleaner.clean_batch(texts)
    
    assert len(cleaned) == 2  # Only 2 should pass
    assert all(item["text"] is not None for item in cleaned)
    assert all(len(item["text"]) >= 10 for item in cleaned)


def test_line_ending_normalization():
    """Test that different line endings are normalized."""
    cleaner = TextCleaner(normalize_whitespace=True, min_text_length=10)
    
    # Windows line endings
    text_crlf = "Line 1\r\nLine 2\r\nLine 3"
    cleaned_crlf, _ = cleaner.clean(text_crlf)
    
    # Unix line endings
    text_lf = "Line 1\nLine 2\nLine 3"
    cleaned_lf, _ = cleaner.clean(text_lf)
    
    # Mac line endings
    text_cr = "Line 1\rLine 2\rLine 3"
    cleaned_cr, _ = cleaner.clean(text_cr)
    
    # All should result in the same normalized text
    assert cleaned_crlf == cleaned_lf
    assert cleaned_lf == cleaned_cr


def test_strip_project_gutenberg_boilerplate():
    """Test that Project Gutenberg header/footer are removed (e.g. pg100)."""
    cleaner = TextCleaner(remove_headers=False, remove_footers=False, min_text_length=1)
    text = (
        "The Project Gutenberg eBook of Hamlet\n"
        "Title: Hamlet\n"
        "Author: William Shakespeare\n\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK HAMLET ***\n\n"
        "ACT I. Scene I.\n"
        "HAMLET: To be or not to be.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK HAMLET ***\n\n"
        "End of Project Gutenberg."
    )
    cleaned, stats = cleaner.clean(text)
    assert cleaned is not None
    assert "*** START OF" not in cleaned
    assert "*** END OF" not in cleaned
    assert "Project Gutenberg eBook" not in cleaned
    assert "End of Project Gutenberg" not in cleaned
    assert "ACT I. Scene I." in cleaned
    assert "HAMLET: To be or not to be." in cleaned
    assert "pg_boilerplate_removal" in stats["operations_applied"]
