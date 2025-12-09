from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

IN_PATH = Path("data/processed/label_bio_3.jsonl")
OUT_TRAIN = Path("data/processed/bio_train.jsonl")
OUT_VAL   = Path("data/processed/bio_val.jsonl")
OUT_TEST  = Path("data/processed/bio_test.jsonl")


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)   # expects columns: text, label (0/1/2)


def write_jsonl(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {"text": row["text"], "label": int(row["label"])}
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(df)} rows → {path}")


def main():
    df = load_jsonl(IN_PATH)
    print("Total rows:", len(df))
    print("Label counts:", df["label"].value_counts())

    # 80% train, 20% temp
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    # split 20% temp into 10% val / 10% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["label"],
    )

    write_jsonl(train_df, OUT_TRAIN)
    write_jsonl(val_df, OUT_VAL)
    write_jsonl(test_df, OUT_TEST)


if __name__ == "__main__":
    main()
