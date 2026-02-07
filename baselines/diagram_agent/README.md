# DiagramAgent Baseline

This baseline uses [DiagramAgent](https://huggingface.co/DiagramAgent/Diagram_to_Code_Agent), a Qwen2-VL-7B-based model fine-tuned for diagram-to-code generation. Given an architecture diagram image, it generates Graphviz DOT code representing the diagram structure.

## Setup

1. Create and activate the conda environment:

```bash
conda env create -f diag.yaml
conda activate diagagent
```

2. Obtain a [Hugging Face API key](https://huggingface.co/settings/tokens) with access to the DiagramAgent model.

## Usage

```bash
python diag_eval.py --input <input.tsv> --output <output.tsv> --hf_key <your_hf_key>
```

**Arguments:**
- `--input` : Path to the input TSV file (must contain an `Image_URL` column)
- `--output` : Path to the output TSV file (will include a `predicted_code` column)
- `--hf_key` : Hugging Face API key for model access

## Notes

- The model uses `device_map="auto"` and will utilize available GPUs automatically.
- Results are written incrementally to the output file to preserve progress on long runs.
- Each row is appended immediately after processing, so partial results are available if the run is interrupted.
