# Data Extraction and Cleaning - Usage Guide

## Overview

The data extraction and cleaning pipeline processes RTF files from the `training_materials/` directory, extracts plain text, and cleans it for LLM training.

## Quick Start

### 1. Install Dependencies

Make sure you have the required dependencies installed:

```bash
pip install striprtf pyyaml
```

Or install all dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

**Basic usage (dry run - shows statistics only):**

```bash
python scripts/process_data.py --dry-run
```

**With custom configuration:**

```bash
# First, copy the example config
cp configs/data_config.yaml.example configs/data_config.yaml

# Edit configs/data_config.yaml as needed, then run:
python scripts/process_data.py --config configs/data_config.yaml
```

**With custom input directory:**

```bash
python scripts/process_data.py --input-dir /path/to/rtf/files --dry-run
```

**Balanced dataset (oversample Hamletmachine, then clean):**

Ensure Shakespeare text is in `training_materials/` (e.g. `pg100.txt`). Then either set `use_balanced_dataset: true` in config or run:

```bash
python scripts/process_data.py --use-balanced-dataset --config configs/data_config.yaml
```

This runs `scripts/build_balanced_dataset.py` first (writes weighted copies to `data/staged/`), then runs the pipeline on `data/staged/` so the train split has roughly 20× more Hamletmachine exposure than Shakespeare.

**Build balanced dataset only (no cleaning):**

```bash
python scripts/build_balanced_dataset.py --config configs/data_config.yaml
```

## Command Line Options

```
--config PATH           Path to data configuration YAML file (default: use defaults)
--input-dir PATH        Input directory with RTF files (default: training_materials)
--output-dir PATH      Output directory for processed data (default: data/processed)
--file-patterns PATTERNS  File patterns (default: from config)
--use-balanced-dataset Run build_balanced_dataset first, then process from data/staged
--dry-run               Run without saving output (just show statistics)
```

## Configuration

The pipeline uses configuration from `configs/data_config.yaml` (or defaults if not specified).

Key configuration options:

```yaml
input:
  raw_data_dir: "training_materials"  # Directory with RTF files
  file_patterns: ["*.rtf", "*.txt"]
  use_balanced_dataset: false         # If true, run build_balanced_dataset first
  sources:                            # Used when use_balanced_dataset is true
    - path: "Hamletmachine_ENG.txt"
      weight: 20
    - path: "pg100.txt"
      weight: 1

processing:
  remove_headers: true                # Remove headers from text
  remove_footers: true                # Remove footers from text
  normalize_whitespace: true          # Normalize whitespace
  min_text_length: 50                 # Minimum characters per text
  max_text_length: 10000              # Maximum characters per text
  encoding: "utf-8"                   # File encoding
  handle_encoding_errors: "replace"   # How to handle encoding errors
```

## What It Does

### Step 1: Extraction
- Reads RTF files from the input directory
- Extracts plain text using the `striprtf` library
- Tracks metadata (source file, file size, text length)

### Step 2: Cleaning
- Removes headers (URLs, copyright notices, etc.)
- Removes footers (page numbers, etc.)
- Normalizes whitespace (multiple spaces → single space, line breaks)
- Filters text by length (removes too short, truncates too long)
- Handles encoding errors gracefully

### Output
Currently, the pipeline outputs cleaned text data in memory. The next step (Milestone 2, FR-3) will format this into Hugging Face Dataset format and save it.

## Example Output

```
2026-01-19 23:46:15,522 - INFO - Starting data extraction and cleaning pipeline
2026-01-19 23:46:15,522 - INFO - Input directory: training_materials
2026-01-19 23:46:15,522 - INFO - File patterns: ['*.rtf']
2026-01-19 23:46:15,522 - INFO - Step 1: Extracting text from RTF files...
2026-01-19 23:46:15,523 - INFO - Found 4 files matching patterns ['*.rtf']
2026-01-19 23:46:16,439 - INFO - Successfully extracted 1397551 characters from DasKapital_ENG.rtf
2026-01-19 23:46:16,556 - INFO - Successfully extracted 186090 characters from Hamlet_Gutenberg.rtf
2026-01-19 23:46:16,698 - INFO - Successfully extracted 175119 characters from Hamlet_MIT.rtf
2026-01-19 23:46:16,839 - INFO - Successfully extracted 12946 characters from Hamletmachine_ENG.rtf
2026-01-19 23:46:16,839 - INFO - Extraction complete: 4/4 files successful
2026-01-19 23:46:16,839 - INFO - Step 2: Cleaning extracted text...
2026-01-19 23:46:17,072 - INFO - Cleaned batch: 4/4 texts passed (0 filtered)

Processing Statistics
============================================================
Total texts processed: 4
Total characters: 40,000
Average text length: 10000 characters

Texts by source file:
  DasKapital_ENG.rtf: 1 texts, 10,000 characters
  Hamlet_Gutenberg.rtf: 1 texts, 10,000 characters
  Hamlet_MIT.rtf: 1 texts, 10,000 characters
  Hamletmachine_ENG.rtf: 1 texts, 10,000 characters
```

## Programmatic Usage

You can also use the pipeline programmatically:

```python
from pathlib import Path
from hamletmachine.data import DataProcessingPipeline

# Create pipeline with default config
pipeline = DataProcessingPipeline()

# Run extraction and cleaning
cleaned_data = pipeline.run_extraction_and_cleaning()

# Access cleaned texts
for item in cleaned_data:
    print(f"Source: {item['source_file']}")
    print(f"Text length: {len(item['text'])}")
    print(f"Text preview: {item['text'][:100]}...")
```

Or with a custom configuration:

```python
from hamletmachine.data import DataProcessingPipeline, load_data_config

# Load config from file
config = load_data_config(Path("configs/data_config.yaml"))

# Create pipeline
pipeline = DataProcessingPipeline(config=config)

# Run pipeline
cleaned_data = pipeline.run_extraction_and_cleaning()
```

## Testing

Run the test suite:

```bash
# Test extraction
pytest tests/test_data_extraction.py -v

# Test cleaning
pytest tests/test_data_cleaning.py -v

# Test both
pytest tests/test_data_extraction.py tests/test_data_cleaning.py -v
```

## Troubleshooting

### ModuleNotFoundError: No module named 'striprtf'

Install the missing dependency:
```bash
pip install striprtf
```

### No files found

Check that:
1. The input directory exists
2. RTF files match the file patterns in config
3. You have read permissions for the files

### Text is being truncated

The `max_text_length` setting in config limits text length. For longer texts, you'll need to implement chunking (coming in FR-3: Data Formatting).

### Encoding errors

Try adjusting `handle_encoding_errors` in config:
- `"replace"` - Replace problematic characters (default)
- `"ignore"` - Ignore problematic characters
- `"strict"` - Raise errors (not recommended)

## Next Steps

After extraction and cleaning, the next steps are:
1. **Data Formatting (FR-3)**: Convert to Hugging Face Dataset format
2. **Tokenization Setup (FR-4)**: Configure tokenizer
3. **Dataset Splitting (FR-5)**: Split into train/val/test sets
