# Dataset Curation Pipeline

This module implements the dataset curation pipeline for **Text2Arch**, which extracts structured graph representations of ML architecture diagrams from research papers. The pipeline takes raw paper images and PDFs as input and produces labeled directed graphs (DOT format) paired with descriptive text extracted from the papers.

## Pipeline Overview

```
Images ──► annotate.py ──► ocr_with_arrowheads.py ──► merge_flowchart_ocr.py ──► merged_flowchart_ocr_to_graph.py
                                                                                          │
PDFs ──► text_extraction.py ──────────────────────────────────────────────────────► Final Dataset
```

## Scripts

### 1. `annotate.py` — Flowchart Element Detection

Runs a Detectron2 Faster R-CNN model on architecture diagram images to detect five element types: **Node**, **Text**, **Arrow**, **Fig Label**, and **Node Label**. Outputs per-image JSON files with bounding boxes, confidence scores, and class predictions.

```bash
python annotate.py \
    --frcnn_weights /path/to/model_weights.pth \
    --test_images_dir /path/to/images/ \
    --output_dir /path/to/visualizations/ \
    --json_output_dir /path/to/detection_jsons/
```

### 2. `ocr_with_arrowheads.py` — OCR and Arrow Detection

Uses Microsoft Florence-2 for OCR with region detection, combined with OpenCV contour analysis for arrow endpoint detection. Processes a folder of images and produces a consolidated JSON file.

```bash
python ocr_with_arrowheads.py \
    --input_folder /path/to/images/ \
    --output_json /path/to/ocr_results.json
```

### 3. `merge_flowchart_ocr.py` — Merge Detection + OCR

Spatially associates OCR text regions with detected flowchart elements (from steps 1 and 2). Text is assigned to nodes via center-point containment; unmatched text is assigned to the nearest bounding box by Euclidean distance.

```bash
python merge_flowchart_ocr.py \
    --image_folder_path /path/to/images/ \
    --json_folder_path /path/to/detection_jsons/ \
    --ocr_json_folder /path/to/ocr_jsons/ \
    --output_json_folder /path/to/merged_jsons/
```

### 4. `merged_flowchart_ocr_to_graph.py` — Graph Generation

Converts merged detection+OCR JSON files into directed graphs in DOT format. Arrow connectivity is determined by finding the nearest labeled nodes to each arrow's head and tail endpoints.

```bash
python merged_flowchart_ocr_to_graph.py \
    --json_folder /path/to/merged_jsons/ \
    --tsv_path /path/to/image_urls.tsv \
    --output_dot_folder /path/to/dot_files/
```

### 5. `text_extraction.py` — Paper Text Extraction

Downloads arXiv PDFs, extracts text surrounding figure references, and computes TF-IDF cosine similarity against figure captions and OCR text to identify the most relevant descriptive passages. Supports parallel processing.

```bash
python text_extraction.py \
    --path /path/to/images/ \
    --json_path /path/to/figure_metadata.json \
    --json_ocr_path /path/to/florence_ocr_jsons/ \
    --pdf_cache_dir ./pdf_cache \
    --output_json_dir /path/to/text_output/
```

## Dependencies

- **Python** >= 3.8
- **detectron2** — Object detection (used by `annotate.py`)
- **transformers** — Hugging Face Transformers for Florence-2 OCR (used by `ocr_with_arrowheads.py`)
- **torch** — PyTorch backend
- **opencv-python** (`cv2`) — Image processing and arrow detection
- **Pillow** (`PIL`) — Image loading
- **numpy** — Numerical operations
- **scipy** — Spatial distance computation (used by `merged_flowchart_ocr_to_graph.py`)
- **scikit-learn** — TF-IDF and cosine similarity (used by `text_extraction.py`)
- **pypdf** — PDF text extraction (used by `text_extraction.py`)
- **requests** — HTTP downloads (arXiv PDFs, images)
- **tqdm** — Progress bars

Install core dependencies:

```bash
pip install torch torchvision detectron2 transformers opencv-python Pillow numpy scipy scikit-learn pypdf requests tqdm
```
