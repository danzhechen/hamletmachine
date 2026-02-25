# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Hamletmachine LLM is a Python ML pipeline for training a GPT-2-based language model on literary texts. It is a single Python package (`hamletmachine-llm`) with a `src/` layout. There are no microservices, Docker, or databases — everything runs as local Python scripts.

### Environment

- **Python 3.11** is required (see `.python-version`). The VM has it installed at `/usr/bin/python3.11` with a virtualenv at `/workspace/venv`.
- Activate with `source /workspace/venv/bin/activate`.
- The `hamletmachine` package is not installed in editable mode due to a `pyproject.toml` validation error (empty `email` field in `project.authors`). Instead, set `PYTHONPATH=/workspace/src:$PYTHONPATH` before running any Python commands that import `hamletmachine`.

### Key commands

All commands assume the venv is activated and `PYTHONPATH=/workspace/src:$PYTHONPATH` is set.

- **Lint**: `black --check src/ tests/ scripts/`, `flake8 src/ tests/`, `mypy src/`
- **Test**: `pytest tests/ -v`
- **Data pipeline**: `python scripts/process_data.py --config configs/data_config.yaml`
- **Balanced data pipeline**: `python scripts/process_data.py --use-balanced-dataset --config configs/data_config.yaml`

See `README.md` for full documentation.

### Known issues

- **Test failures are pre-existing**: 6 of 11 tests in `test_data_cleaning.py` fail because the tests were written against an older API (different default values, operation names, and header/footer removal behavior). The implementation (`cleaner.py`) has since changed. These are not environment issues.
- **mypy reports errors** about missing type stubs (`types-PyYAML`, `datasets`, `striprtf`) and a syntax error in vendored torch code. These are expected and do not affect runtime.
- **flake8 reports many style warnings** (E501 line length, W293 whitespace). The codebase uses `black` with `line-length = 100` which differs from flake8's default of 79.
