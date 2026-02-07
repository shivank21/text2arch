# Architecture Diagram Classifier

Binary classifier that distinguishes **architecture diagrams** from other scientific figures. Fine-tunes a pretrained vision model on labeled image datasets and evaluates with standard classification metrics.

## Supported Models

| Model | Flag | Pretrained Checkpoint |
|-------|------|-----------------------|
| ViT | `vit` | `google/vit-base-patch16-224` |
| BEiT | `beit` | `microsoft/beit-base-patch16-224` |
| CLIP | `clip` | `openai/clip-vit-base-patch32` |
| ResNet-50 | `resnet` | `microsoft/resnet-50` |

## Training

`arch_classifier.py` fine-tunes a pretrained model for binary classification (architecture vs. non-architecture). It supports combining multiple data sources: SciFig, SciFig-Pilot, Paper2Fig, and GAN-generated synthetic images. Classes are automatically balanced before training.

### Example

```bash
python arch_classifier.py \
    --model_type vit \
    --positive_folder_scifig data/SciFig/archpng \
    --negative_folder_scifig data/SciFig/noarchpng \
    --scifig_pilot_png data/SciFig-pilot/png \
    --dataset_inclusion scifig+pilot \
    --epochs 10 \
    --batch_size 64 \
    --lr 5e-5 \
    --checkpoint_path checkpoints \
    --use_augmentation
```

### Key Training Arguments

| Argument | Description |
|----------|-------------|
| `--model_type` | Model architecture: `vit`, `beit`, `clip`, or `resnet` |
| `--positive_folder_scifig` | **(required)** Path to SciFig architecture images |
| `--negative_folder_scifig` | **(required)** Path to SciFig non-architecture images |
| `--scifig_pilot_png` | Path to SciFig-Pilot png folder (category subfolders) |
| `--paper2fig_train` | Path to Paper2Fig folder (`Arch_Images/` and `nonArch_Images/` subfolders) |
| `--gan_arch_images` | Path to GAN-generated positive images |
| `--gan_nonarch_images` | Path to GAN-generated negative images |
| `--dataset_inclusion` | `scifig+pilot` (default) or `scifig` |
| `--use_augmentation` | Enable random rotation, flip, and crop augmentation |
| `--checkpoint_path` | Directory for saving best checkpoints |
| `--expt_prefix` | Prefix for checkpoint filenames |
| `--epochs` | Number of training epochs (default: 10) |
| `--lr` | Learning rate (default: 5e-5) |
| `--batch_size` | Batch size (default: 64) |

## Evaluation

`test_classifier.py` loads a trained checkpoint and evaluates on a held-out test set. Reports accuracy, precision, recall, F1 score, true negative rate, and confusion matrix.

### Example

```bash
python test_classifier.py \
    --model_type vit \
    --positive_folder data/Paper2Fig/Arch_Images \
    --negative_folder data/Paper2Fig/nonArch_Images \
    --checkpoint_path checkpoints/model_vit_5.pth \
    --save_prediction 1 \
    --prediction_path predictions.csv
```

### Key Evaluation Arguments

| Argument | Description |
|----------|-------------|
| `--model_type` | Must match the architecture used for training |
| `--positive_folder` | **(required)** Path to positive test images |
| `--negative_folder` | **(required)** Path to negative test images |
| `--checkpoint_path` | **(required)** Path to the `.pth` checkpoint file |
| `--save_prediction` | Save per-image predictions to CSV (1=yes, 0=no) |
| `--prediction_path` | Output CSV path (default: `predictions.csv`) |
