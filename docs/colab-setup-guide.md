# Google Colab Setup Guide

This guide walks you through setting up and running hamletmachine LLM training on Google Colab's free GPU.

## Why Colab?

- **Free GPU access**: T4 GPU (16GB VRAM) on free tier
- **🎓 Colab Pro (Free for Students)**: Better GPUs (P100, V100), longer sessions (24h), more RAM
- **No setup required**: Everything runs in the browser
- **Easy sharing**: Share notebooks with collaborators
- **Google Drive integration**: Save checkpoints to Drive
- **Auto-optimization**: Notebook automatically detects your GPU and optimizes settings

## Prerequisites

1. **Google Account**: Sign in to [Google Colab](https://colab.research.google.com)
2. **Processed Dataset**: Your training data should be processed and ready (see data pipeline docs)
3. **Project Files**: Either on GitHub or ready to upload

## Quick Start

### Option 1: Use the Colab Notebook Template (Recommended)

1. **Open the notebook**:
   - Upload `notebooks/train_on_colab.ipynb` to Google Colab
   - Or open it directly if your repo is on GitHub

2. **Enable GPU**:
   - Runtime → Change runtime type
   - Hardware accelerator → **GPU (T4)**
   - Click Save

3. **Run all cells** in order:
   - The notebook will guide you through each step

### Option 2: Manual Setup

1. **Create a new Colab notebook**

2. **Enable GPU**:
   - Runtime → Change runtime type → GPU (T4)

3. **Mount Google Drive** (optional, for saving checkpoints):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Upload or clone your project**:
   ```python
   # Option A: Clone from GitHub
   !git clone https://github.com/yourusername/hamletmachine.git
   %cd hamletmachine
   
   # Option B: Upload manually
   # Use the file browser on the left sidebar
   ```

5. **Run setup script**:
   ```python
   !bash scripts/setup_colab.sh
   # Or
   !python scripts/setup_colab.py
   ```

6. **Upload your processed dataset**:
   - Upload `data/processed/train.jsonl`
   - Upload `data/processed/validation.jsonl`
   - Upload `data/processed/test.jsonl`

7. **Start training**:
   ```python
   !python scripts/train.py --config configs/train_config.yaml
   ```

## Detailed Steps

### Step 1: Enable GPU

1. Click **Runtime** in the menu bar
2. Select **Change runtime type**
3. Under **Hardware accelerator**, select **GPU**
4. Click **Save**

**Verify GPU is enabled**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Step 2: Mount Google Drive (Optional)

Mounting Google Drive allows you to:
- Save checkpoints that persist after session ends
- Access files from your Drive
- Share checkpoints between sessions

```python
from google.colab import drive
drive.mount('/content/drive')
```

After running this, you'll need to:
1. Click the authorization link
2. Sign in to your Google account
3. Copy the authorization code
4. Paste it in the notebook

**Set checkpoint directory to Drive**:
```python
CHECKPOINT_DIR = '/content/drive/MyDrive/hamletmachine/checkpoints'
import os
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
```

### Step 3: Get Your Project Code

#### Option A: Clone from GitHub (Recommended)

If your project is on GitHub:

```python
!git clone https://github.com/yourusername/hamletmachine.git
%cd hamletmachine
```

#### Option B: Upload Manually

1. Click the **folder icon** on the left sidebar
2. Click **Upload** button
3. Upload your project folder (or zip and extract)
4. Navigate to the project:
   ```python
   %cd /content/hamletmachine
   ```

#### Option C: Upload from Google Drive

If you've uploaded your project to Drive:

```python
# Copy from Drive to Colab
!cp -r /content/drive/MyDrive/hamletmachine /content/
%cd /content/hamletmachine
```

### Step 4: Install Dependencies

Run the setup script:

```python
!bash scripts/setup_colab.sh
```

Or install manually:

```python
!pip install -q transformers datasets accelerate tokenizers torch numpy pandas pyyaml tensorboard tqdm
```

### Step 5: Upload Your Dataset

You need your processed dataset files:
- `data/processed/train.jsonl`
- `data/processed/validation.jsonl`
- `data/processed/test.jsonl`

**Option A: Upload via file browser**
1. Click folder icon on left
2. Navigate to `data/processed/`
3. Upload the JSONL files

**Option B: Upload from Drive**
```python
!cp /content/drive/MyDrive/hamletmachine/data/processed/*.jsonl /content/hamletmachine/data/processed/
```

**Option C: Process on Colab** (if you have raw materials):
```python
from hamletmachine.data.pipeline import DataPipeline
pipeline = DataPipeline(config_path='configs/data_config.yaml')
pipeline.run()
```

### Step 6: Configure Training

Create or edit `configs/train_config.yaml`:

```yaml
model:
  architecture: "gpt2"  # Start with small model

training:
  output_dir: "/content/drive/MyDrive/hamletmachine/checkpoints"  # Or /content/models/checkpoints
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-5
  fp16: true  # Enable for T4 GPU

data:
  train_file: "data/processed/train.jsonl"
  validation_file: "data/processed/validation.jsonl"
  max_seq_length: 1024
```

### Step 7: Start Training

**Using the training script**:
```python
!python scripts/train.py --config configs/train_config.yaml
```

**Or use the notebook cells** (if using the template notebook):
- Run all cells in order
- Training will start automatically

### Step 8: Monitor Training

**TensorBoard** (included in notebook):
```python
%load_ext tensorboard
%tensorboard --logdir logs --port 6006
```

**Or check logs**:
```python
!tail -f logs/training.log
```

## Colab Limitations & Tips

### Free Tier Limitations

- **Session timeout**: ~9-12 hours (may disconnect earlier)
- **Idle timeout**: 90 minutes of inactivity
- **GPU availability**: May not always get GPU (especially during peak hours)
- **RAM**: Limited (may need to reduce batch size)

### 🎓 Colab Pro Benefits (Free for Students)

If you have Colab Pro (free for students), you get:

- **Better GPUs**: P100 (16GB) or V100 (16GB/32GB) instead of just T4
- **Longer Sessions**: Up to 24 hours (vs ~9-12 hours on free tier)
- **More RAM**: Better for larger models and datasets
- **Better Availability**: Priority GPU access
- **Auto-Optimization**: The notebook automatically detects your GPU and optimizes:
  - Model size (small on T4, medium/large on P100/V100)
  - Batch sizes (larger on better GPUs)
  - Training configuration

**The notebook will automatically detect your GPU type and optimize settings!**

### Tips for Success

1. **Save frequently**: 
   - Use Google Drive for checkpoints
   - Save intermediate results

2. **Handle disconnections**:
   - Enable checkpoint saving (`save_steps: 500`)
   - Resume from checkpoint if disconnected

3. **Optimize for Colab**:
   - Use smaller batch sizes (4-8)
   - Enable FP16 (`fp16: true`)
   - Use gradient accumulation for larger effective batch size

4. **Monitor resources**:
   ```python
   # Check GPU memory
   !nvidia-smi
   
   # Check RAM
   !free -h
   ```

5. **Download results**:
   - If not using Drive, download checkpoints before session ends
   - Use the download cell in the notebook

### Colab Pro ($9.99/month)

Consider upgrading if you need:
- Longer sessions (up to 24 hours)
- Better GPU availability
- More RAM
- Priority access

## Troubleshooting

### GPU Not Available

**Problem**: `torch.cuda.is_available()` returns `False`

**Solutions**:
1. Check Runtime → Change runtime type → GPU is selected
2. Wait a few minutes and try again (GPU may be temporarily unavailable)
3. Consider Colab Pro for better availability

### Out of Memory

**Problem**: CUDA out of memory error

**Solutions**:
1. Reduce batch size: `per_device_train_batch_size: 2`
2. Increase gradient accumulation: `gradient_accumulation_steps: 8`
3. Enable FP16: `fp16: true`
4. Reduce sequence length: `max_seq_length: 512`

### Session Disconnected

**Problem**: Training stopped due to disconnection

**Solutions**:
1. Resume from checkpoint:
   ```python
   !python scripts/train.py --config configs/train_config.yaml --resume_from_checkpoint models/checkpoints/checkpoint-500
   ```
2. Use Google Drive for checkpoints (they persist)
3. Consider Colab Pro for longer sessions

### Import Errors

**Problem**: Module not found errors

**Solutions**:
1. Make sure you're in the project directory: `%cd /content/hamletmachine`
2. Add to Python path:
   ```python
   import sys
   sys.path.insert(0, '/content/hamletmachine')
   ```
3. Install the package: `!pip install -e .`

## Next Steps

After training completes:

1. **Download the model** (if not on Drive):
   ```python
   from google.colab import files
   files.download('models/checkpoints/final_model')
   ```

2. **Evaluate the model**:
   ```python
   !python scripts/evaluate.py --model models/checkpoints/final_model
   ```

3. **Test inference**:
   ```python
   !python scripts/inference.py --model models/checkpoints/final_model --text "Your prompt"
   ```

## Additional Resources

- [Colab Documentation](https://colab.research.google.com/notebooks/intro.ipynb)
- [Colab GPU FAQ](https://research.google.com/colaboratory/faq.html)
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)
- [Project Training Guide](./training-stage-plan.md)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the training logs
3. Check TensorBoard for training curves
4. Verify your dataset format matches expectations
