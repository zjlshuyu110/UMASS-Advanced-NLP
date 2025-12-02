from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset


PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_TRAIN = PROCESSED_DIR / "biomed_shekswess_train.jsonl"
OUT_VAL   = PROCESSED_DIR / "biomed_shekswess_val.jsonl"


def write_jsonl(df: pd.DataFrame, path: Path):
    """Save DataFrame with columns ['text', 'label'] to JSONL."""
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = {
                "text": str(row["text"]).strip(),
                "label": str(row["label"]).strip(),
            }
            f.write(json.dumps(rec) + "\n")
    print(f"✅ Wrote {len(df)} rows to {path}")


def load_shekswess_sentiment() -> pd.DataFrame:
    """
    Load Shekswess/ai-healthcare-biomedical-sentiment from HF.

    Expected columns (from HF viewer):
      - 'text'
      - 'label'  (strings: 'positive', 'negative', 'neutral')
      - plus extra columns we ignore: topic, domain, language, etc.
    """
    print("📥 Loading Shekswess/ai-healthcare-biomedical-sentiment...")
    ds = load_dataset("Shekswess/ai-healthcare-biomedical-sentiment")

    train_split = ds["train"]
    df = train_split.to_pandas()
    print("Columns in Shekswess dataset:", list(df.columns))

    if "prompt" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected 'text' and 'label' columns in Shekswess dataset.")

    df = df.dropna(subset=["prompt", "label"]).copy()
    df["text"] = df["prompt"].astype(str)
    df["label"] = df["label"].astype(str)

    # Keep only canonical sentiment labels, including NEUTRAL
    allowed = {"negative", "neutral", "positive"}
    df = df[df["label"].isin(allowed)]

    df = df[["text", "label"]]

    print(f"Shekswess sentiment: {len(df)} rows after cleaning.")
    print("Label distribution:\n", df["label"].value_counts())
    return df


def main(test_size: float = 0.1, random_state: int = 42):
    df = load_shekswess_sentiment()

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    write_jsonl(train_df, OUT_TRAIN)
    write_jsonl(val_df, OUT_VAL)
    print("🎉 Finished preparing Shekswess biomedical sentiment data for SFT.")


if __name__ == "__main__":
    main()
