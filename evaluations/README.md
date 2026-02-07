# Evaluation Metrics

This module provides evaluation scripts for **Text2Arch**, measuring the quality of generated architecture diagrams against ground truth across three complementary dimensions: text similarity, graph structure, and image fidelity.

## Metrics Overview

### Text-Based Metrics
- **ROUGE-L** -- Longest common subsequence overlap between DOT code strings.
- **chrF** -- Character n-gram F-score.
- **Edit Distance** -- Levenshtein distance between predicted and reference code.
- **CodeBLEU** -- Code-aware BLEU variant incorporating syntax and dataflow.
- **Pass@1** -- Exact match rate between predicted and reference code.
- **RUBY** -- Jaccard similarity over unique tokens.

### Graph-Based Metrics
- **Node Precision / Recall / F1** -- Similarity-weighted node matching using the Hungarian algorithm.
- **Node PR-AUC** -- Area under the node precision-recall curve across similarity thresholds.
- **Edge Precision / Recall / F1** -- Edge correctness after mapping predicted nodes to ground truth.
- **Edge Jaccard** -- Jaccard similarity over edge sets.
- **Edge PR-AUC** -- Area under the edge precision-recall curve.

### Image-Based Metrics
- **FID** -- Frechet Inception Distance.
- **KID** -- Kernel Inception Distance.
- **CLIP-FID** -- FID computed with CLIP features.
- **LPIPS** -- Learned Perceptual Image Patch Similarity.
- **PSNR** -- Peak Signal-to-Noise Ratio.
- **MS-SSIM** -- Multi-Scale Structural Similarity Index.

## Scripts

| Script | Description |
|---|---|
| `text_metrics.py` | Combined text and graph metrics for DOT files referenced in a TSV. |
| `diagram_agent_eval.py` | Text-only metrics from a TSV with inline DOT code columns. |
| `graph_metrics.py` | Graph-structural metrics for Graphviz DOT files. |
| `graph_metrics_tikz.py` | Graph-structural metrics for TikZ format files. |
| `image_metric.py` | Image similarity metrics between two folders of PNG diagrams. |

## Installation

```bash
pip install -r req.txt
```

**Note:** `pygraphviz` requires the Graphviz system library. On Ubuntu/Debian:
```bash
sudo apt-get install graphviz libgraphviz-dev
```

## Usage

### Text + Graph Metrics (DOT files from folders)

```bash
python text_metrics.py \
    --gt_folder path/to/gt_dot_files/ \
    --pred_folder path/to/pred_dot_files/ \
    --test_tsv_file path/to/test.tsv \
    -o results.csv \
    --verbose
```

Use `--text_only` to skip graph metrics. Use `--threshold` to adjust the node matching similarity threshold (default: 0.5).

### Text Metrics (DOT code in TSV)

```bash
python diagram_agent_eval.py \
    --input_file path/to/predictions.tsv \
    -o results.csv \
    --verbose
```

The TSV must contain `Dot code` (ground truth) and `predicted_code` columns.

### Graph Metrics (DOT format)

```bash
python graph_metrics.py \
    --gt_folder path/to/gt_dot_files/ \
    --pred_folder path/to/pred_dot_files/ \
    --test_tsv_file path/to/test.tsv \
    --output results.csv \
    --threshold 0.3
```

To evaluate a single file pair:
```bash
python graph_metrics.py \
    --gt_folder path/to/gt/ \
    --pred_folder path/to/pred/ \
    --test_tsv_file path/to/test.tsv \
    --single_file example.dot
```

### Graph Metrics (TikZ format)

```bash
python graph_metrics_tikz.py \
    --gt_folder path/to/gt_tikz_files/ \
    --pred_folder path/to/pred_tikz_files/ \
    --test_tsv_file path/to/test.tsv \
    --output results.csv \
    --directed
```

### Image Metrics

```bash
python image_metric.py \
    --folder1 path/to/gt_images/ \
    --folder2 path/to/pred_images/ \
    --output metrics.txt
```

Both folders must contain the same number of PNG files with matching filenames.
