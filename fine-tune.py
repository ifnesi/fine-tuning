#!/usr/bin/env python3
"""
Fine-tune Gemma 3 270M on a MacBook Pro M1 using Transformers + TRL.

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

Example:
    python fine-tune.py \
        --model google/gemma-3-270m-it \
        --dataset dataset/nist-rmf.jsonl \
        --output-dir models/gemma-3-270m-nist \
        --max-seq-length 2048
"""

import os
import sys
import torch
import logging
import argparse

from typing import Dict
from functools import partial

from trl import SFTConfig, SFTTrainer
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma 3 270M on Apple Silicon with Transformers + TRL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-270m-it",
        help="Model name or local path to fine-tune.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/nist-rmf.jsonl",
        help="Path to the training dataset in JSONL format. Each row should contain prompt and response fields.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/gemma-3-270m-nist",
        help="Directory where the fine-tuned model and tokenizer will be saved.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum training sequence length in tokens. On M1, reduce to 1024 or 512 if you hit out-of-memory errors.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs over the dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device training batch size. Batch size 1 is safer on Apple Silicon.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps used to simulate a larger effective batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for supervised fine-tuning.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="How often to emit training logs.",
    )
    parser.add_argument(
        "--save-strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
        help="Checkpoint save strategy.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="How often to save checkpoints if using step-based checkpointing.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token. Overrides HF_TOKEN from the environment or .env file if provided.",
    )

    return parser.parse_args()


def load_environment(hf_token_override: str | None) -> str | None:
    load_dotenv()
    return hf_token_override or os.environ.get("HF_TOKEN")


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_example(example: Dict[str, str], tokenizer) -> Dict[str, str]:
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def prepare_dataset(dataset_path: str, tokenizer):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    logger.info("Loading dataset from %s", dataset_path)
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    logger.info("Loaded %s examples", len(dataset))

    dataset = dataset.map(
        partial(format_example, tokenizer=tokenizer),
        remove_columns=dataset.column_names,
    )
    return dataset


def load_model_and_tokenizer(model_name: str, hf_token: str | None, device: str):
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
    )

    logger.info("Loading model: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.float32 if device == "mps" else "auto",
    )

    model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def main() -> None:
    args = parse_args()
    hf_token = load_environment(args.hf_token)
    device = get_device()

    logger.info("=" * 80)
    logger.info("Starting fine-tuning")
    logger.info("Model: %s", args.model)
    logger.info("Dataset: %s", args.dataset)
    logger.info("Device: %s", device)
    logger.info("Max sequence length: %s", args.max_seq_length)
    logger.info("=" * 80)

    if device == "mps":
        logger.info("Using Apple Silicon MPS backend")
        logger.info("If you hit memory errors, reduce --max-seq-length to 1024 or 512")
        logger.info("Also consider lowering epochs or keeping batch size at 1")

    model, tokenizer = load_model_and_tokenizer(args.model, hf_token, device)
    dataset = prepare_dataset(args.dataset, tokenizer)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        fp16=False,
        bf16=False,
        loss_type="nll",
        completion_only_loss=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training")
    trainer.train()

    logger.info("Saving model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Done")


if __name__ == "__main__":
    main()