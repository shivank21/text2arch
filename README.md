# Text2Arch: Generating Architecture Diagrams from Text Descriptions

Text2Arch is a pipeline for automatically generating architecture diagrams from natural language descriptions of ML systems. Given a textual description of a model's architecture, the system produces a structured graph representation in Graphviz DOT format.

## Repository Structure

```
text2arch/
├── arch_classifier/       # Binary classifier for architecture diagram detection
├── baselines/             # Baseline methods (AutomaTikZ, DiagramAgent)
├── dataset_curation/      # Dataset construction pipeline (OCR, text extraction, graph generation)
├── evaluations/           # Evaluation metrics (text, graph, image similarity)
├── model/                 # Model training (DeepSpeed) and inference scripts
└── prompts/               # Prompt templates for DOT code generation
```

## Modules

### Architecture Classifier (`arch_classifier/`)

Binary image classifier that distinguishes architecture diagrams from other scientific figures. Supports ViT, BEiT, CLIP, and ResNet-50 backbones. Used as a preprocessing step to filter architecture diagrams from paper figure datasets.

### Dataset Curation (`dataset_curation/`)

End-to-end pipeline for constructing the training dataset:
1. **Flowchart element detection** using Faster R-CNN (Detectron2)
2. **OCR** using Microsoft Florence-2 with arrow endpoint detection
3. **Merging** detected elements with OCR text via spatial association
4. **Graph generation** converting merged annotations into DOT format
5. **Text extraction** from arXiv PDFs matched to figures via TF-IDF similarity

### Model Training & Inference (`model/`)

Fine-tunes language models to convert text descriptions into DOT graph code. Uses DeepSpeed for distributed multi-GPU training. Includes inference scripts for both base and instruction-tuned models with few-shot prompting.

**Tested models:** LLaMA-3-8B, DeepSeek-7B-Chat, Qwen2-7B-Instruct

### Baselines (`baselines/`)

Two baseline methods for comparison:
- **AutomaTikZ** — Text-to-TikZ generation using a fine-tuned LLaMA model
- **DiagramAgent** — Image-to-DOT generation using Qwen2-VL-7B

### Evaluations (`evaluations/`)

Comprehensive evaluation suite across three dimensions:
- **Text metrics:** ROUGE-L, chrF, Edit Distance, CodeBLEU, Pass@1, RUBY
- **Graph metrics:** Node/Edge Precision, Recall, F1, Jaccard (via Hungarian matching)
- **Image metrics:** FID, KID, CLIP-FID, LPIPS, PSNR, MS-SSIM

### Prompts (`prompts/`)

Prompt templates used for DOT code generation, TikZ-to-DOT conversion, and description comparison tasks.

## Getting Started

### Prerequisites

- Python >= 3.8
- CUDA-capable GPU (required for training and most inference)
- Conda (recommended for environment management)

### Installation

```bash
git clone <repository-url>
cd text2arch

# Install core dependencies
pip install torch torchvision transformers datasets accelerate deepspeed
pip install pandas numpy scipy scikit-learn tqdm
```

Each module has additional dependencies documented in its own README. See the individual subdirectory READMEs for detailed setup and usage instructions.

### Quick Start

**Train a model:**
```bash
cd model/
deepspeed --num_gpus=8 model_training_deepspeed.py \
    --model_name meta-llama/Meta-Llama-3-8B \
    --hf_key <HF_TOKEN> \
    --wandb_key <WANDB_TOKEN> \
    --train_file ../data/train.tsv \
    --val_file ../data/val.tsv \
    --test_file ../data/test.tsv \
    --save_path ./checkpoints/llama3-8b \
    --ds_config /path/to/deepspeed.json
```

**Run inference:**
```bash
python model/infer_instruct.py \
    --model_path ./checkpoints/llama3-8b \
    --test_file ./data/test.tsv \
    --output_file ./results/predictions.tsv
```

**Evaluate results:**
```bash
python evaluations/text_metrics.py \
    --gt_folder ./data/gt_dot/ \
    --pred_folder ./results/pred_dot/ \
    --test_tsv_file ./data/test.tsv \
    -o results.csv
```

## Dataset

The dataset is available on Hugging Face: [shivank21/text2archdata](https://huggingface.co/datasets/shivank21/text2archdata)

It contains the following files:
- **dot2.tsv** — Labeled data generated using the dot2 pipeline
- **dot3.tsv** — Labeled data generated using the dot3 pipeline
- **newTest.tsv** — Test set
- **manual.tsv** — Manually annotated test set

## Our Trained Checkpoints

- **DeepSeek:** [shivank21/text2arch-deepseek](https://huggingface.co/shivank21/text2arch-deepseek)
- **Qwen:** [shivank21/text2arch-qwen](https://huggingface.co/shivank21/text2arch-qwen)
- **Llama:** [shivank21/text2arch-llama](https://huggingface.co/shivank21/text2arch-llama)
