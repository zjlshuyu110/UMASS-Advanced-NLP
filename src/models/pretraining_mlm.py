import argparse
import json
from pathlib import Path



import yaml
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Domain-adaptive MLM training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., src/configs/mlm_biomed.yaml)",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)



def load_mlm_dataset(train_files, tokenizer, max_seq_length):
    """
    train_files: list of JSONL files, each with a 'text' field.
    Returns a tokenized dataset ready for MLM.
    """
    # Load all JSONL files and concatenate into one dataset
    datasets = []
    for file in train_files:
        print(f"Loading {file} ...")
        ds = load_dataset("json", data_files=file, split="train")
        datasets.append(ds)

    if len(datasets) == 1:
        raw_dataset = datasets[0]
    else:
        raw_dataset = concatenate_datasets(datasets)

    print("Total samples:", len(raw_dataset))

    # Tokenize text
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )


    tokenized = raw_dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=1,
        remove_columns=raw_dataset.column_names,  # ⬅ drop *all* original columns, including "text"
    )   


    # Group into chunks for MLM (continuous segments up to max_seq_length)
    def group_texts(examples):
        # Concatenate all token lists
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])

        # Drop remainder
        total_length = (total_length // max_seq_length) * max_seq_length

        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated.items()
        }
        return result

    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        num_proc=1,
    )
    

    print("MLM dataset size (number of chunks):", len(lm_dataset))
    return lm_dataset


def main():
    args = parse_args()
    config = load_config(args.config)

    model_name_or_path = config["model_name_or_path"]
    train_files = config["train_files"]
    output_dir = config["output_dir"]
    max_seq_length = config.get("max_seq_length", 128)
    mlm_probability = config.get("mlm_probability", 0.15)

    # Load tokenizer and model
    print(f"Loading tokenizer and model from {model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)

    # Load dataset
    train_dataset = load_mlm_dataset(train_files, tokenizer, max_seq_length)

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=int(config.get("max_steps", 1000)),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 8)),
        learning_rate=float(config.get("learning_rate", 5e-5)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        warmup_steps=int(config.get("warmup_steps", 0)),
        logging_steps=int(config.get("logging_steps", 50)),
        save_steps=int(config.get("save_steps", 500)),
        save_total_limit=int(config.get("save_total_limit", 2)),
        prediction_loss_only=True,
        report_to=[],
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("Starting MLM training ...")
    trainer.train()
    print("Training complete. Saving model ...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save config used
    cfg_out = Path(output_dir) / "mlm_config_used.json"
    with cfg_out.open("w") as f:
        json.dump(config, f, indent=2)
    print("Saved config to", cfg_out)


if __name__ == "__main__":
    main()
