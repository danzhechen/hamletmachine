#!/usr/bin/env python3
"""
Data processing script.

This script runs the data extraction and cleaning pipeline on training materials.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hamletmachine.data import DataProcessingPipeline, load_data_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the data processing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process training materials: extract and clean text"
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to data configuration YAML file (default: use defaults)'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=None,
        help='Input directory with RTF files (default: from config)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for processed data (default: from config)'
    )
    parser.add_argument(
        '--file-patterns',
        nargs='+',
        default=None,
        help='File patterns to match (e.g., "*.txt" "*.rtf"). Default: from config'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without saving output (just show statistics)'
    )
    parser.add_argument(
        '--no-split',
        action='store_true',
        help='Skip dataset splitting (save full dataset only)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        if not args.config.exists():
            logger.error(f"Configuration file not found: {args.config}")
            return 1
        config = load_data_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        logger.info("Using default configuration")
        config = None
    
    # Create pipeline
    if config:
        pipeline = DataProcessingPipeline(config=config)
    else:
        pipeline = DataProcessingPipeline()
    
    # Run full pipeline (extraction -> cleaning -> formatting -> splitting -> saving)
    try:
        save_dataset = not args.dry_run
        result = pipeline.run_full_pipeline(
            input_dir=args.input_dir,
            file_patterns=args.file_patterns,
            save_dataset=save_dataset,
            split_dataset=not args.no_split
        )
        
        cleaned_data = result.get('cleaned_data', [])
        if not cleaned_data:
            logger.error("No data was processed successfully")
            return 1
        
        # Print statistics
        logger.info("\n" + "="*60)
        logger.info("Processing Statistics")
        logger.info("="*60)
        
        total_texts = len(cleaned_data)
        total_chars = sum(len(item['text']) for item in cleaned_data)
        avg_chars = total_chars / total_texts if total_texts > 0 else 0
        
        # Group by source file
        by_source = {}
        for item in cleaned_data:
            source = item['source_file']
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(item)
        
        logger.info(f"Total texts processed: {total_texts}")
        logger.info(f"Total characters: {total_chars:,}")
        logger.info(f"Average text length: {avg_chars:.0f} characters")
        logger.info(f"\nTexts by source file:")
        for source, items in sorted(by_source.items()):
            chars = sum(len(item['text']) for item in items)
            logger.info(f"  {source}: {len(items)} texts, {chars:,} characters")
        
        # Handle dataset statistics
        if not args.no_split and 'statistics' in result:
            # Split statistics already printed by splitter
            logger.info("\nSplit statistics shown above.")
        else:
            dataset = result.get('dataset')
            if dataset is not None:
                logger.info(f"\nDataset Statistics:")
                logger.info(f"  Total chunks: {len(dataset)}")
                logger.info(f"  Dataset features: {list(dataset.features.keys())}")
                if 'token_count' in dataset.features:
                    total_tokens = sum(dataset['token_count'])
                    avg_tokens = total_tokens / len(dataset) if len(dataset) > 0 else 0
                    logger.info(f"  Total tokens: {total_tokens:,}")
                    logger.info(f"  Average tokens per chunk: {avg_tokens:.0f}")
        
        if args.dry_run:
            logger.info("\nDry run mode: No output saved")
        else:
            saved_paths = result.get('saved_paths', {})
            if saved_paths:
                logger.info(f"\nSaved files:")
                for name, path in saved_paths.items():
                    logger.info(f"  {name}: {path}")
            else:
                logger.info(f"\nOutput directory: {pipeline.formatter.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
