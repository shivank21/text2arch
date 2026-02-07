import os
import random
import argparse
import logging

import numpy as np
import torch
import torch.distributed as dist
import wandb
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
from huggingface_hub import login, HfApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chat_template(dataset, tokenizer):
    """Apply a custom chat template to format training examples.

    Each example is formatted as a system/user/assistant conversation where
    the user provides a text description and the assistant responds with
    DOT graph code.
    """
    chats = []

    template_str = (
        "{%- for message in messages %}"
        "{%- if message['role'] == 'system' %}## System:{{- message['content'].strip() }}\n"
        "{%- elif message['role'] == 'user' %}## Instruction: {{- message['content'].strip() }}\n"
        "{%- elif message['role'] == 'assistant' %}## Response: {{- message['content'].strip() }}\n"
        "{%- endif %}"
        "{%- endfor %}"
    )
    tokenizer.chat_template = template_str

    for sample in dataset:
        user = sample['Cleaned Description']
        assistant = sample['Dot code']
        chat = [
            {
                "role": "system",
                "content": (
                    "You are an expert in analyzing technical descriptions of system architecture, "
                    "workflows, and process pipelines, and a code design specialist skilled in "
                    "graph visualization using DOT language."
                )
            },
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        eos = "<|eot_id|>"
        chat = tokenizer.apply_chat_template(chat, tokenize=False) + eos
        chats.append(chat)

    return dataset.add_column("text", chats), tokenizer


def load_local_tsv_datasets(train_path, val_path, test_path):
    """Load train/validation/test splits from TSV files."""
    try:
        def load_split(path):
            df = pd.read_csv(path, sep='\t', dtype=str)
            return Dataset.from_pandas(df)

        return DatasetDict({
            "train": load_split(train_path),
            "validation": load_split(val_path),
            "test": load_split(test_path)
        })
    except Exception as e:
        logger.error(f"Failed to load TSV datasets: {e}")
        raise


def main(args):
    """Main training pipeline."""
    set_seed(42)

    # Initialize distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    # Disable flash attention for compatibility
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        logger.info("Flash Attention disabled.")
    except Exception as e:
        logger.warning(f"Could not disable flash attention: {e}")

    # Authenticate with external services
    try:
        login(args.hf_key)
        wandb.login(key=args.wandb_key)
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_eos_token = True
        tokenizer.add_bos_token = True
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

    # Load and preprocess dataset
    try:
        dataset = load_local_tsv_datasets(args.train_file, args.val_file, args.test_file)
        train_dataset, tokenizer = chat_template(dataset["train"], tokenizer)
        eval_dataset, tokenizer = chat_template(dataset["validation"], tokenizer)

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=1024)

        train_tokenized = train_dataset.map(tokenize_function, batched=True)
        train_tokenized = train_tokenized.remove_columns(['Cleaned Description', 'Dot code', 'text'])
        eval_tokenized = eval_dataset.map(tokenize_function, batched=True)
        eval_tokenized = eval_tokenized.remove_columns(['Cleaned Description', 'Dot code', 'text'])
    except Exception as e:
        logger.error(f"Error during tokenization or preprocessing: {e}")
        raise

    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            offload_folder=args.offload_folder,
            torch_dtype=torch.bfloat16,
        )
        model.config.use_cache = False

        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Configure training arguments
    training_arguments = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=50,
        learning_rate=5e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=250,
        do_eval=True,
        deepspeed=args.ds_config,
    )

    # Train
    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            args=training_arguments,
        )
        trainer.train()
        trainer.save_model()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Save or push model
    try:
        if args.push_to_hub:
            logger.info("Pushing model and tokenizer to Hugging Face Hub.")
            base_model_name = args.model_name.split('/')[-1]
            hub_repo_name = f"text2arch_{base_model_name}"
            api = HfApi()
            api.create_repo(repo_id=hub_repo_name, exist_ok=True)
            model.push_to_hub(hub_repo_name)
            tokenizer.push_to_hub(hub_repo_name)
            logger.info(f"Model and tokenizer pushed to Hub: {hub_repo_name}")
        else:
            logger.info("Saving model and tokenizer locally.")
            os.makedirs(args.save_path, exist_ok=True)
            model.save_pretrained(args.save_path)
            tokenizer.save_pretrained(args.save_path)
            logger.info(f"Model and tokenizer saved to {args.save_path}")
    except Exception as e:
        logger.error(f"Failed to save model/tokenizer: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model for Text2Arch with DeepSpeed")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--hf_key", type=str, required=True, help="Hugging Face API key")
    parser.add_argument("--wandb_key", type=str, required=True, help="Weights & Biases API key")
    parser.add_argument("--save_path", type=str, required=True, help="Local path to save the trained model")
    parser.add_argument("--train_file", type=str, required=True, help="Path to train.tsv")
    parser.add_argument("--val_file", type=str, required=True, help="Path to val.tsv")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test.tsv")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--offload_folder", type=str, default=None, help="Folder for CPU offloading of weight shards")
    parser.add_argument("--push_to_hub", action="store_true", default=False, help="Push model to HuggingFace Hub instead of saving locally")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (set by DeepSpeed)")
    parser.add_argument("--ds_config", type=str, required=True, help="Path to DeepSpeed config JSON (must be absolute)")

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logger.critical(f"Pipeline crashed: {e}")
