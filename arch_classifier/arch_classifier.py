import argparse
import os
import random

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import (
    AdamW,
    AutoModelForImageClassification,
    BeitForImageClassification,
    CLIPForImageClassification,
    ViTForImageClassification,
)
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a vision model for architecture diagram classification."
    )
    parser.add_argument(
        "--model_type", type=str, default="vit",
        choices=["vit", "beit", "clip", "resnet"],
        help="Model architecture to fine-tune.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--positive_folder_scifig", type=str, required=True,
        help="Path to positive (architecture) images from SciFig dataset.",
    )
    parser.add_argument(
        "--negative_folder_scifig", type=str, required=True,
        help="Path to negative (non-architecture) images from SciFig dataset.",
    )
    parser.add_argument(
        "--scifig_pilot_png", type=str, default="",
        help="Path to SciFig-Pilot png folder (contains category subfolders).",
    )
    parser.add_argument(
        "--paper2fig_train", type=str, default="",
        help="Path to Paper2Fig folder for training (expects Arch_Images/ and nonArch_Images/ subfolders).",
    )
    parser.add_argument(
        "--train_valid_split", type=float, default=0.2,
        help="Fraction of data reserved for validation + test.",
    )
    parser.add_argument(
        "--valid_test_split", type=float, default=0.5,
        help="Fraction of held-out data used for test (vs. validation).",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default="checkpoints",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--expt_prefix", type=str, default="",
        help="Prefix for checkpoint filenames: model_{prefix}_{epoch}.pth",
    )
    parser.add_argument(
        "--dataset_inclusion", type=str, default="scifig+pilot",
        choices=["scifig+pilot", "scifig"],
        help="Which dataset combination to train on.",
    )
    parser.add_argument(
        "--gan_arch_images", type=str, default="",
        help="Path to GAN-generated architecture images (positive).",
    )
    parser.add_argument(
        "--gan_nonarch_images", type=str, default="",
        help="Path to GAN-generated non-architecture images (negative).",
    )
    parser.add_argument(
        "--use_augmentation", action="store_true",
        help="Apply data augmentation (random rotation, flip, crop) during training.",
    )
    return parser.parse_args()


class CustomDataset(torch.utils.data.Dataset):
    """Wraps a HuggingFace Dataset for use with PyTorch DataLoader."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        label = 1 if item["new_label"] == "positive" else 0
        if self.transform:
            image = self.transform(image)
        return image, label


def create_data_loader(dataset, batch_size, transform, shuffle=True):
    """Create a DataLoader from a HuggingFace Dataset."""
    custom_dataset = CustomDataset(dataset, transform=transform)
    return DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)


def load_image_folder(folder, label):
    """Load all images from a folder and assign a label."""
    return [
        {"image": os.path.join(folder, img), "new_label": label}
        for img in os.listdir(folder)
    ]


def build_transform(use_augmentation):
    """Build image transform pipeline."""
    if use_augmentation:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def load_model(model_type):
    """Load a pretrained model for binary classification."""
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
    transform = build_transform(args.use_augmentation)

    # --- Collect training samples ---
    samples = []

    # SciFig dataset (always included)
    samples += load_image_folder(args.positive_folder_scifig, "positive")
    samples += load_image_folder(args.negative_folder_scifig, "negative")

    # SciFig-Pilot dataset
    if args.dataset_inclusion == "scifig+pilot" and args.scifig_pilot_png:
        pilot_path = args.scifig_pilot_png
        pos_categories = ["architecture diagram", "neural networks"]
        for cat in pos_categories:
            cat_path = os.path.join(pilot_path, cat)
            if os.path.isdir(cat_path):
                samples += load_image_folder(cat_path, "positive")
        for cat in os.listdir(pilot_path):
            if cat not in pos_categories and os.path.isdir(os.path.join(pilot_path, cat)):
                samples += load_image_folder(os.path.join(pilot_path, cat), "negative")

    # GAN-generated synthetic data
    if args.gan_arch_images:
        samples += load_image_folder(args.gan_arch_images, "positive")
    if args.gan_nonarch_images:
        samples += load_image_folder(args.gan_nonarch_images, "negative")

    # Paper2Fig dataset
    if args.paper2fig_train:
        samples += load_image_folder(os.path.join(args.paper2fig_train, "Arch_Images"), "positive")
        samples += load_image_folder(os.path.join(args.paper2fig_train, "nonArch_Images"), "negative")

    # --- Split into train / validation / test ---
    random.shuffle(samples)
    full_dataset = Dataset.from_list(samples)

    train_test_split = full_dataset.train_test_split(test_size=args.train_valid_split)
    train_dataset = train_test_split["train"]
    val_test_split = train_test_split["test"].train_test_split(test_size=args.valid_test_split)
    validation_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]

    split_dataset = DatasetDict({
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    })

    # --- Balance classes per split ---
    positive_examples = split_dataset.filter(lambda x: x["new_label"] == "positive")
    negative_examples = split_dataset.filter(lambda x: x["new_label"] == "negative")

    balanced_dataset = {}
    for split in split_dataset.keys():
        min_count = min(len(positive_examples[split]), len(negative_examples[split]))
        pos_samples = positive_examples[split].shuffle(seed=42).select(range(min_count))
        neg_samples = negative_examples[split].shuffle(seed=42).select(range(min_count))
        balanced_dataset[split] = concatenate_datasets([pos_samples, neg_samples])

    print(f"Balanced dataset sizes - Train: {len(balanced_dataset['train'])}, "
          f"Val: {len(balanced_dataset['validation'])}, Test: {len(balanced_dataset['test'])}")

    # --- Create data loaders ---
    train_loader = create_data_loader(balanced_dataset["train"], args.batch_size, transform, shuffle=True)
    val_loader = create_data_loader(balanced_dataset["validation"], args.batch_size, transform, shuffle=False)
    test_loader = create_data_loader(balanced_dataset["test"], args.batch_size, transform, shuffle=False)

    # --- Model setup ---
    model = load_model(args.model_type)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    expt_prefix = args.expt_prefix if args.expt_prefix else args.model_type
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # --- Training loop ---
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "epoch_loss": f"{total_loss / len(progress_bar):.4f}",
            })

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, labels=labels)
                val_loss += outputs.loss.item()
                _, predicted = outputs.logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * correct / total
        print(f"Epoch {epoch + 1}: Val Loss={val_loss / len(val_loader):.4f}, Val Acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_file = os.path.join(args.checkpoint_path, f"model_{expt_prefix}_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_file)
            print(f"  Saved checkpoint: {ckpt_file}")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")

    # --- Test ---
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.logits.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    print(f"Test Accuracy: {100.0 * test_correct / test_total:.2f}%")


if __name__ == "__main__":
    main()
