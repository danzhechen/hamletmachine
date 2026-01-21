# Hamletmachine LLM

A lightweight Language Model (LLM) training project for the Hamletmachine play, built with the Hugging Face ecosystem.

## Project Overview

This project implements a complete end-to-end LLM training pipeline, from data preparation through model training to deployment. The model is trained on Hamletmachine-related texts and other literary materials.

## Features

- **Complete Training Pipeline**: Data preparation → Model training → Evaluation → Deployment
- **Hugging Face Integration**: Built on Transformers, Datasets, Accelerate, and Tokenizers
- **Flexible Architecture**: Support for training from scratch or fine-tuning pretrained models
- **Comprehensive Tooling**: Data processing, training scripts, evaluation metrics, and inference API

## Project Structure

```
hamletmachine/
├── src/
│   └── hamletmachine/
│       ├── data/          # Data processing pipeline
│       ├── models/        # Model architecture and configuration
│       ├── training/      # Training scripts and utilities
│       ├── evaluation/    # Evaluation metrics and scripts
│       ├── inference/     # Inference and serving
│       └── utils/         # Shared utilities
├── scripts/               # Standalone scripts
├── data/                  # Data directories
│   ├── raw/              # Raw training materials
│   ├── processed/        # Processed datasets
│   └── cache/            # Cached data
├── models/               # Model checkpoints and saved models
│   ├── checkpoints/      # Training checkpoints
│   └── pretrained/       # Pretrained models
├── training_materials/   # Original RTF training files
├── notebooks/            # Jupyter notebooks for exploration
├── tests/                # Test suite
├── docs/                 # Documentation
└── configs/              # Configuration files
```

## Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for training) OR Google Colab (free GPU)
- Git

### Installation Options

#### Option 1: Local Installation

1. **Clone the repository** (if using Git):
   ```bash
   git clone <repository-url>
   cd hamletmachine
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch; import transformers; print('Installation successful!')"
   ```

#### Option 2: Google Colab (Recommended for Free GPU Training)

Train on Google Colab's free T4 GPU without local setup:

1. **Open the Colab notebook**:
   - Upload `notebooks/train_on_colab.ipynb` to [Google Colab](https://colab.research.google.com)
   - Or open directly if your repo is on GitHub

2. **Enable GPU**:
   - Runtime → Change runtime type → Hardware accelerator → **GPU (T4)**

3. **Follow the notebook cells** - everything is automated!

📖 **See [Colab Setup Guide](./docs/colab-setup-guide.md) for detailed instructions.**

## Quick Start

### 1. Data Preparation

Process the training materials:
```bash
python scripts/prepare_data.py --input training_materials/ --output data/processed/
```

### 2. Train Model

Start training (configuration TBD):
```bash
python scripts/train.py --config configs/train_config.yaml
```

### 3. Evaluate Model

Evaluate the trained model:
```bash
python scripts/evaluate.py --model models/checkpoints/best_model --data data/processed/test
```

### 4. Inference

Run inference with the trained model:
```bash
python scripts/inference.py --model models/checkpoints/best_model --text "Your prompt here"
```

## Development Status

**Current Milestone**: Milestone 1 - Project Setup ✅

**Upcoming Milestones**:
- Milestone 2: Data Pipeline Development
- Milestone 3: Model Architecture Selection & Configuration
- Milestone 4: Training Infrastructure Setup
- Milestone 5-10: Training, Evaluation, Deployment (see full plan in brainstorming session)

## Configuration

Key decision points to configure:

- **Model Size**: Small (125M), Medium (350M), or Larger
- **Training Strategy**: From scratch or fine-tune pretrained model
- **Hardware**: Single GPU, multi-GPU, or cloud
- **Use Case**: Determines evaluation metrics

## Documentation

- [Colab Setup Guide](./docs/colab-setup-guide.md) - **Train on Google Colab for free GPU access**
- [Cloud GPU Options](./docs/cloud-gpu-options.md) - Cost analysis and cloud platform comparison
- [Training Stage Plan](./docs/training-stage-plan.md) - Detailed training implementation plan
- [Brainstorming Session](./_bmad-output/analysis/brainstorming-session-2026-01-19T16:34:03.md) - Complete implementation plan with 10 milestones
- [Training Materials](./training_materials/) - Source RTF files

## Contributing

This is a research/educational project. Contributions and suggestions are welcome!

## License

MIT License

## Acknowledgments

- Hugging Face for the excellent Transformers ecosystem
- Training materials: Hamletmachine, Hamlet, Das Kapital texts
