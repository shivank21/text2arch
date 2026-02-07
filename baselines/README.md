# Baselines

This directory contains the baseline methods evaluated in the Text2Arch paper. Each subdirectory is self-contained with its own environment configuration and run script.

## Included Baselines

### AutomaTikZ

**Directory:** `automatikz/`

AutomaTikZ (Belouadi & Eger, 2024) is a text-to-TikZ generation model. We use the `nllg/tikz-clima-13b` checkpoint, a fine-tuned LLaMA model that generates TikZ code from natural language descriptions. This baseline represents the text-to-diagram approach using LaTeX TikZ as the output format.

- **Input:** Text descriptions (TSV with `Cleaned Description` column)
- **Output:** TikZ code (`predicted_tikz` column)
- **Reference:** https://github.com/potamides/AutomaTikZ

### DiagramAgent

**Directory:** `diagram_agent/`

DiagramAgent is a Qwen2-VL-7B-based model fine-tuned for diagram-to-code generation. Given an architecture diagram image, it generates Graphviz DOT code. This baseline represents the image-to-code approach, converting visual diagrams directly into structured graph descriptions.

- **Input:** Diagram images (TSV with `Image_URL` column)
- **Output:** Graphviz DOT code (`predicted_code` column)
- **Reference:** https://huggingface.co/DiagramAgent/Diagram_to_Code_Agent

## Quick Start

Each baseline has its own conda environment. See the README in each subdirectory for detailed setup and usage instructions.

```bash
# AutomaTikZ
cd automatikz/
conda env create -f env.yaml
conda activate automatikz

# DiagramAgent
cd diagram_agent/
conda env create -f diag.yaml
conda activate diagagent
```
