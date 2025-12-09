from pathlib import Path
import json
import random


def load_jsonl(path):
    """Load JSONL into a list of dicts."""
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(records, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"✅ Wrote {len(records)} examples to {out_path}")


def combine_datasets(input_files, output_file, add_source=False, shuffle=True, seed=42):
    """
    input_files: list of paths
    output_file: single output path
    add_source: if True, add a 'source' field indicating which file it came from
    """
    all_records = []

    for path in input_files:
        path = Path(path)
        print(f"📥 Loading {path} ...")
        recs = load_jsonl(path)

        if add_source:
            for r in recs:
                r["source"] = path.stem  # e.g. 'biomed_drug_reviews_train'

        all_records.extend(recs)
        print(f"  Added {len(recs)} records from {path}")

    if shuffle:
        random.seed(seed)
        random.shuffle(all_records)
        print(f"🔀 Shuffled {len(all_records)} total records")

    save_jsonl(all_records, output_file)


if __name__ == "__main__":
    # Example usage – adjust paths to match your repo

    

    # --- Finance train combined ---
    finance_inputs = [
        "data/processed/label_financial_2.jsonl",
        "data/processed/label_financial_3.jsonl",
    ]
    finance_out = "data/processed/finance_combined.jsonl"
    combine_datasets(finance_inputs, finance_out, add_source=True)
