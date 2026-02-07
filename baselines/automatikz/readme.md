# AutomaTikZ Baseline

This baseline uses [AutomaTikZ](https://github.com/potamides/AutomaTikZ) (Belouadi & Eger, 2024) to generate TikZ code from text descriptions. Specifically, it uses the `nllg/tikz-clima-13b` model, a fine-tuned LLaMA variant trained on TikZ documents.

## Setup

1. Clone the AutomaTikZ repository inside this directory:

```bash
git clone https://github.com/potamides/AutomaTikZ.git
```

2. Create and activate the conda environment:

```bash
conda env create -f env.yaml
conda activate automatikz
```

3. Copy `run.py` into the cloned `AutomaTikZ/` directory:

```bash
cp run.py AutomaTikZ/
```

## Usage

```bash
cd AutomaTikZ
python run.py --input <input.tsv> --output <output.tsv>
```

**Arguments:**
- `--input` : Path to the input TSV file (must contain a `Cleaned Description` column)
- `--output` : Path to the output TSV file (will include a `predicted_tikz` column)
- `--batch_save` : Save intermediate results every N entries (default: 50)

## Notes

- The model requires approximately 29 GB of GPU memory.
- Processing the full dataset may take a significant amount of time (~70 hours on a single GPU).
- Intermediate results are saved periodically to avoid losing progress.
