import argparse
import os
import random

import pandas as pd
import torch
from datasets import Dataset
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import (
    AutoModelForImageClassification,
    BeitForImageClassification,
    CLIPForImageClassification,
    ViTForImageClassification,
)
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained architecture diagram classifier on a test set."
    )
    parser.add_argument(
        "--model_type", type=str, default="vit",
        choices=["vit", "beit", "clip", "resnet"],
        help="Model architecture (must match the checkpoint).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--positive_folder", type=str, required=True,
        help="Path to folder containing positive (architecture) test images.",
    )
    parser.add_argument(
        "--negative_folder", type=str, required=True,
        help="Path to folder containing negative (non-architecture) test images.",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to the model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--save_prediction", type=int, default=1, choices=[0, 1],
        help="Whether to save per-image predictions to CSV (1=yes, 0=no).",
    )
    parser.add_argument(
        "--prediction_path", type=str, default="predictions.csv",
        help="Output path for the predictions CSV file.",
    )
    return parser.parse_args()


class CustomDataset(torch.utils.data.Dataset):
    """Wraps a HuggingFace Dataset for inference, returning image path alongside data."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        path = image
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        label = 1 if item["new_label"] == "positive" else 0
        if self.transform:
            image = self.transform(image)
        return image, label, path


def create_data_loader(dataset, batch_size, transform):
    """Create a DataLoader from a HuggingFace Dataset."""
    custom_dataset = CustomDataset(dataset, transform=transform)
    return DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)


def load_model(model_type):
    """Load a pretrained model architecture for binary classification."""
    model_configs = {
        "beit": ("microsoft/beit-base-patch16-224", BeitForImageClassification),
        "vit": ("google/vit-base-patch16-224", ViTForImageClassification),
        "clip": ("openai/clip-vit-base-patch32", CLIPForImageClassification),
        "resnet": ("microsoft/resnet-50", AutoModelForImageClassification),
    }
    pretrained_name, model_class = model_configs[model_type]
    return model_class.from_pretrained(
        pretrained_name, num_labels=2, ignore_mismatched_sizes=True
    )


def main():
    args = parse_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load test images
    positive_images = [
        {"image": os.path.join(args.positive_folder, img), "new_label": "positive"}
        for img in os.listdir(args.positive_folder)
    ]
    negative_images = [
        {"image": os.path.join(args.negative_folder, img), "new_label": "negative"}
        for img in os.listdir(args.negative_folder)
    ]

    all_images = positive_images + negative_images
    random.shuffle(all_images)
    test_dataset = Dataset.from_list(all_images)
    test_loader = create_data_loader(test_dataset, args.batch_size, transform)

    # Load model and checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_type)
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    all_filenames = []

    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc="Evaluating"):
            filenames = [os.path.basename(p) for p in paths]
            all_filenames.extend(filenames)

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.logits.max(1)

            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Report metrics
    accuracy = 100.0 * test_correct / test_total
    precision = precision_score(all_labels, all_predictions, average="binary")
    recall = recall_score(all_labels, all_predictions, average="binary")
    f1 = f1_score(all_labels, all_predictions, average="binary")
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()
    tnr = tn / (tn + fp)

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Precision:     {precision:.2f}")
    print(f"Recall:        {recall:.2f}")
    print(f"F1 Score:      {f1:.2f}")
    print(f"True Neg Rate: {tnr:.2f}")
    print(f"\nConfusion Matrix:\n{cm}")

    # Save predictions
    if args.save_prediction == 1:
        preds_df = pd.DataFrame({
            "image": all_filenames,
            "predicted": all_predictions,
            "label": all_labels,
        })
        preds_df.to_csv(args.prediction_path, index=False)
        print(f"Predictions saved to {args.prediction_path}")


if __name__ == "__main__":
    main()
