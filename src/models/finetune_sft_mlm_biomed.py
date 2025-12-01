import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_jsonl_datasets(files):
    """Load one or more JSONL files into a single HF Dataset."""
    if isinstance(files, str):
        files = [files]
    datasets = [load_dataset("json", data_files=f, split="train") for f in files]
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    model_path = config["model_name_or_path"]
    output_dir = config["output_dir"]
    num_labels = int(config["num_labels"])
    label_map_path = config["label_map"]

    # -------- tokenizer + base model ----------
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
    )

    # -------- optional LoRA adapters ----------
    use_lora = bool(config.get("use_lora", False))
    if use_lora:
        lora_cfg = LoraConfig(
            task_type="SEQ_CLS",
            r=int(config.get("lora_r", 8)),
            lora_alpha=int(config.get("lora_alpha", 16)),
            lora_dropout=float(config.get("lora_dropout", 0.1)),
        )
        model = get_peft_model(base_model, lora_cfg)
    else:
        model = base_model

    # -------- load labeled data ----------
    train_files = config["train_files"]
    eval_files = config["eval_files"]

    train_ds = load_jsonl_datasets(train_files)
    eval_ds = load_jsonl_datasets(eval_files)

    with open(label_map_path, "r") as f:
        label_map = json.load(f)  # e.g. {"negative": 0, "neutral": 1, "positive": 2}
    label_to_id = {k: int(v) for k, v in label_map.items()}

    def encode(ex):
        text = ex["text"]
        label = label_to_id[ex["label"]]
        tok = tokenizer(text, truncation=True)
        tok["labels"] = label
        return tok

    train_ds = train_ds.map(encode, batched=False)
    eval_ds = eval_ds.map(encode, batched=False)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # -------- training arguments ----------
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=float(config.get("learning_rate", 5e-5)),
        per_device_train_batch_size=int(config.get("batch_size", 8)),
        num_train_epochs=float(config.get("epochs", 3)),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=[],  # no wandb/tensorboard by default
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
