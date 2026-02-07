# Model Training and Inference

This module provides scripts for fine-tuning language models and running inference for the Text2Arch pipeline. The models learn to convert natural language descriptions of system architectures into DOT graph code.

## Contents

| File | Description |
|------|-------------|
| `model_training_deepspeed.py` | Fine-tuning script with DeepSpeed for multi-GPU training |
| `infer_base.py` | Few-shot inference using base (pretrained) models |
| `infer_instruct.py` | Few-shot inference using instruction-tuned models |
| `examples.tsv` | Few-shot examples with pipe-delimited DOT output (used by `infer_instruct.py`) |
| `examples_1.tsv` | Few-shot examples with multi-line DOT output (used by `infer_base.py`) |
| `inference_prompt.txt` | Prompt template for DOT code generation |

## Environment Setup

Create a conda environment with the required dependencies:

```yaml
name: text2arch
channels:
  - defaults
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.11
  - pip
  - pandas=2.2.2
  - numpy=2.0.2
  - pytorch=2.6.0
  - pytorch-cuda=12.4
  - pip:
      - transformers==4.51.3
      - datasets==3.5.0
      - wandb==0.19.9
      - huggingface_hub==0.30.2
      - trl==0.16.1
      - accelerate==1.5.2
```

```bash
conda env create -f environment.yml
conda activate text2arch
```

Verify the installation:

```bash
python -c "import torch; import transformers; import datasets; import wandb; import trl; print('All packages imported successfully.')"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Training with DeepSpeed

Fine-tune a model using DeepSpeed for distributed training across multiple GPUs:

```bash
deepspeed --num_gpus=8 model_training_deepspeed.py \
  --model_name meta-llama/Meta-Llama-3-8B \
  --hf_key <HUGGINGFACE_TOKEN> \
  --wandb_key <WANDB_TOKEN> \
  --train_file ../data/train.tsv \
  --val_file ../data/val.tsv \
  --test_file ../data/test.tsv \
  --batch_size 1 \
  --epochs 5 \
  --save_path ./checkpoints/llama3-8b \
  --ds_config /absolute/path/to/deepspeed.json
```

**Note:** The `--ds_config` path must be absolute due to a known DeepSpeed limitation with relative paths.

### Training Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--model_name` | Yes | HuggingFace model name or local path |
| `--hf_key` | Yes | HuggingFace API token |
| `--wandb_key` | Yes | Weights & Biases API token |
| `--save_path` | Yes | Directory to save the trained model |
| `--train_file` | Yes | Path to training data (TSV) |
| `--val_file` | Yes | Path to validation data (TSV) |
| `--test_file` | Yes | Path to test data (TSV) |
| `--batch_size` | No | Per-device batch size (default: 1) |
| `--epochs` | No | Number of training epochs (default: 5) |
| `--ds_config` | Yes | Absolute path to DeepSpeed config JSON |
| `--offload_folder` | No | Directory for CPU weight offloading |
| `--push_to_hub` | No | Push model to HuggingFace Hub instead of saving locally |

### Supported Models

The training script has been tested with:
- `meta-llama/Meta-Llama-3-8B`
- `deepseek-ai/deepseek-llm-7b-chat`
- `Qwen/Qwen2-7B-Instruct`

## Inference

Two inference scripts are provided depending on the model type.

### Base Models

For pretrained models without instruction tuning. Uses a custom chat template to format few-shot prompts:

```bash
python infer_base.py \
  --model_path ./checkpoints/llama3-8b \
  --test_file ../data/test.tsv \
  --output_file ./results/base_results.tsv \
  --examples_file examples_1.tsv
```

### Instruction-Tuned Models

For instruction-tuned models. Uses the model's native chat template and includes DOT code extraction from free-form responses:

```bash
python infer_instruct.py \
  --model_path ./checkpoints/qwen2-7b-instruct \
  --test_file ../data/test.tsv \
  --output_file ./results/instruct_results.tsv \
  --examples_file examples.tsv
```

### Inference Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--model_path` | Yes | Path to the model (local or HuggingFace) |
| `--test_file` | Yes | Path to test data (TSV with `Cleaned Description` and `Dot code` columns) |
| `--output_file` | Yes | Path to save generated results (TSV) |
| `--examples_file` | No | Path to few-shot examples TSV (falls back to built-in defaults) |

## Data Format

### Training/Test Data

TSV files with at least two columns:
- `Cleaned Description`: Natural language description of the system architecture
- `Dot code`: Ground truth DOT graph code

### Few-Shot Examples

TSV files with two columns:
- `input`: Description text
- `output`: Corresponding DOT code (may use `|` as newline delimiter in `examples.tsv`)
