#!/usr/bin/env python3
"""
Build a balanced training dataset by applying source weights (oversampling).

Reads config (sources with path + weight), loads each source file from
raw_data_dir, and writes repeated copies to data/staged/ so that the
data pipeline sees the desired token distribution (e.g. 20× Hamletmachine).
Run process_data.py with --input-dir data/staged afterward to clean and split.
"""

import sys
import logging
from pathlib import Path

# Project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hamletmachine.data import load_data_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_STAGED_DIR = "data/staged"


def _resolve_files(base_dir: Path, path_pattern: str):
    """Expand path_pattern (file or glob) under base_dir. Returns list of Paths."""
    base_dir = Path(base_dir)
    candidates = list(base_dir.glob(path_pattern))
    return sorted([p for p in candidates if p.is_file()])


def build_balanced_dataset(
    config_path: Path | None = None,
    raw_data_dir: Path | None = None,
    sources: list | None = None,
    staged_dir: Path | None = None,
) -> Path:
    """
    Build staged dataset with weighted source repetition.

    Args:
        config_path: Path to data_config.yaml. If set, raw_data_dir and sources
            are read from config (input.raw_data_dir, input.sources).
        raw_data_dir: Override base dir for source files (default from config).
        sources: Override list of {path, weight} (default from config).
        staged_dir: Output directory (default data/staged).

    Returns:
        Path to staged directory (all sources written as repeated .txt files).
    """
    if config_path and config_path.exists():
        config = load_data_config(config_path)
        input_cfg = config.get("input", {})
        if raw_data_dir is None:
            raw_data_dir = Path(input_cfg.get("raw_data_dir", "training_materials"))
        else:
            raw_data_dir = Path(raw_data_dir)
        if sources is None:
            sources = input_cfg.get("sources", [])
    else:
        raw_data_dir = Path(raw_data_dir or "training_materials")
        sources = sources or []

    # Resolve raw_data_dir relative to project root
    if not raw_data_dir.is_absolute():
        raw_data_dir = PROJECT_ROOT / raw_data_dir
    staged_dir = Path(staged_dir or DEFAULT_STAGED_DIR)
    if not staged_dir.is_absolute():
        staged_dir = PROJECT_ROOT / staged_dir

    staged_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Building balanced dataset: raw_data_dir=%s, staged_dir=%s", raw_data_dir, staged_dir)

    total_written = 0
    for entry in sources:
        path_pattern = entry.get("path", "")
        weight = int(entry.get("weight", 1))
        if not path_pattern or weight < 1:
            continue
        files = _resolve_files(raw_data_dir, path_pattern)
        if not files:
            logger.warning("No files matched pattern: %s under %s", path_pattern, raw_data_dir)
            continue
        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.warning("Skip %s: %s", file_path, e)
                continue
            stem = file_path.stem
            for i in range(weight):
                out_name = f"{stem}_{i + 1:03d}.txt"
                out_path = staged_dir / out_name
                out_path.write_text(text, encoding="utf-8")
                total_written += 1
        logger.info("Source %s (weight %s): %s file(s) -> %s copy/copies each", path_pattern, weight, len(files), weight)

    logger.info("Staged %s file(s) under %s", total_written, staged_dir)
    return staged_dir


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build balanced dataset (weighted source repetition) into data/staged")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "data_config.yaml", help="Data config YAML")
    parser.add_argument("--raw-data-dir", type=Path, default=None, help="Override raw data directory")
    parser.add_argument("--staged-dir", type=Path, default=None, help="Output directory (default: data/staged)")
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        logger.warning("Config not found: %s; using defaults for sources", config_path)
        config_path = None

    build_balanced_dataset(
        config_path=config_path,
        raw_data_dir=args.raw_data_dir,
        staged_dir=args.staged_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
