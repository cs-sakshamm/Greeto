#!/usr/bin/env python3
"""QLoRA fine-tuning for Greeto's Qwen 2.5 admission model.

Defaults are tuned for a single 8 GB GPU (or a Colab T4) and for the
small dataset produced by ``prepare_kb_dataset.py``. The script accepts
Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct or Qwen2.5-3B-Instruct via
``--model_name`` and writes a LoRA adapter that ``export_gguf.py`` can
merge and quantise to Q4_K_M.

This file is intentionally import-guarded: the kiosk runtime does not
need PyTorch, and ``--dry-run`` works without any training dependency.

    python training/train_qwen25.py --dry-run
    python training/train_qwen25.py --model_name Qwen/Qwen2.5-0.5B-Instruct

Training dependencies are listed in training/requirements.txt.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train_file", default="data/qwen25_train.jsonl")
    parser.add_argument("--val_file", default="data/qwen25_val.jsonl")
    parser.add_argument("--output_dir", default="models/qwen25-greeto-lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the JSONL files without importing torch/peft/trl.",
    )
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError("Dataset not found: %s" % path)

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("%s:%d is not valid JSON: %s" % (path, line_number, error))
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                raise ValueError("%s:%d must contain exactly 3 messages" % (path, line_number))
            rows.append(row)

    if not rows:
        raise ValueError("Dataset is empty: %s" % path)

    return rows


def dry_run(args: argparse.Namespace, train: list[dict], val: list[dict]) -> int:
    roles = {message["role"] for row in train for message in row["messages"]}
    kinds = {}
    for row in train + val:
        kind = (row.get("metadata") or {}).get("kind", "unknown")
        kinds[kind] = kinds.get(kind, 0) + 1

    print("Train examples: %d" % len(train))
    print("Val examples:   %d" % len(val))
    print("Roles:          %s" % ", ".join(sorted(roles)))
    print("Kinds:          %s" % kinds)
    print("Model:          %s" % args.model_name)
    print("Output:         %s" % resolve(args.output_dir))
    print("\nFirst example:")
    print(json.dumps(train[0], ensure_ascii=False, indent=2)[:1600])
    print("\nDry run OK. Install training/requirements.txt to train.")
    return 0


def format_example(tokenizer, row: dict) -> dict:
    text = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def build_sft_arguments(args: argparse.Namespace, has_eval: bool):
    common = dict(
        output_dir=str(resolve(args.output_dir)),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        fp16=True,
        report_to=[],
        seed=args.seed,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=args.save_steps if has_eval else None,
    )

    try:
        from trl import SFTConfig

        parameters = inspect.signature(SFTConfig.__init__).parameters
        if "max_length" in parameters:
            common["max_length"] = args.max_seq_len
        elif "max_seq_length" in parameters:
            common["max_seq_length"] = args.max_seq_len

        return SFTConfig(**common), {}

    except ImportError:
        from transformers import TrainingArguments

        return TrainingArguments(**common), {
            "dataset_text_field": "text",
            "max_seq_length": args.max_seq_len,
        }


def main() -> int:
    args = parse_args()

    train_path = resolve(args.train_file)
    val_path = resolve(args.val_file)

    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path) if val_path.exists() else []

    if args.dry_run:
        return dry_run(args, train_rows, val_rows)

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import SFTTrainer
    except ImportError as error:
        print(
            "Training dependencies are missing (%s).\n"
            "Install them with:\n"
            "    python -m pip install -r training/requirements.txt"
            % error
        )
        return 2

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = Dataset.from_list([format_example(tokenizer, row) for row in train_rows])
    val_dataset = (
        Dataset.from_list([format_example(tokenizer, row) for row in val_rows])
        if val_rows
        else None
    )

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }

    if not args.no_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False

    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args, sft_extra = build_sft_arguments(args, has_eval=val_dataset is not None)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora,
        **sft_extra,
    )

    trainer.train()
    trainer.save_model(str(resolve(args.output_dir)))
    tokenizer.save_pretrained(str(resolve(args.output_dir)))

    print("\nLoRA adapter saved to %s" % resolve(args.output_dir))
    print("Next: python training/export_gguf.py --adapter_dir %s" % resolve(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
