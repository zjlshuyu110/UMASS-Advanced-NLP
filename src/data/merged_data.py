from pathlib import Path
import json
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
FILE1 = Path("/Users/user/ANLP_Project/UMASS-Advanced-NLP/data/processed/label_bio_3_val.jsonl")   # <-- replace with your paths
FILE2 = Path("/Users/user/ANLP_Project/UMASS-Advanced-NLP/data/processed/label_financial_3_val.jsonl")   # <-- replace
OUT = Path("data/processed/lable_mixed_3_val.jsonl")


def validate_label(label):
    """Ensure label is numeric (0,1,2)."""
    try:
        label = int(label)
    except:
        raise ValueError(f"Label must be numeric: {label}")

    if label not in {0, 1, 2}:
        raise ValueError(f"Label must be 0,1,2. Got {label}")

    return label


def load_jsonl(path, source_name):
    rows = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text", "").strip()
            label = validate_label(obj.get("label"))

            rows.append({
                "text": text,
                "label": label,
                "source": source_name
            })
    return pd.DataFrame(rows)


def main():
    print("📥 Loading dataset 1...")
    df1 = load_jsonl(FILE1, "dataset1")

    print("📥 Loading dataset 2...")
    df2 = load_jsonl(FILE2, "dataset2")

    print("🔗 Combining...")
    df = pd.concat([df1, df2], ignore_index=True)

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    after = len(df)
    print(f"🧹 Removed {before - after} duplicates. Final size: {after}")

    # Shuffle for training setup
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Write combined jsonl
    with OUT.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({
                "text": row["text"],
                "label": row["label"],
                "source": row["source"]
            }) + "\n")

    print(f"✅ Saved merged dataset with numeric labels to {OUT}")


if __name__ == "__main__":
    main()
