"""
Format as {"text": str, "label": int}

LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

"""

from pathlib import Path
import json
import re
from datasets import load_dataset

RAW_DIR = Path("data/raw/financial")
OUT_PATH = Path("data/processed/label_financial_3.jsonl")

LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def download_investing_if_needed():
    """
    Download financial phrasebank dataset from HuggingFace if not present.
    """
    raw_path = RAW_DIR / "financial_phrasebank.jsonl"
    
    if not raw_path.exists():
        print("Downloading financial_phrasebank from HuggingFace...")
        # Use the parquet URL that works
        ds = load_dataset(
            "parquet",
            data_files="https://huggingface.co/datasets/takala/financial_phrasebank/resolve/4a94a2397c42fc3274b7db5632f913eb914e9e0f/sentences_50agree/financial_phrasebank-train.parquet"
        )
        RAW_DIR.mkdir(parents=True, exist_ok=True)

        with raw_path.open("w", encoding="utf-8") as w:
            for row in ds['train']:
                w.write(json.dumps(row) + "\n")

        print("Saved raw data →", raw_path)
    else:
        print("Financial phrasebank already exists, skipping download.")


def iter_texts_from_phrasebank():
    """
    Read financial phrasebank JSONL and yield normalized text + numeric label.
    """
    raw_path = RAW_DIR / "financial_phrasebank.jsonl"
    
    if raw_path.exists():
        print("Reading:", raw_path)
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Financial phrasebank has 'sentence' and 'label' fields
                text = normalize(obj.get("sentence", ""))
                sentiment = obj.get("label", "")
                
                # Map integer labels to strings
                label_map = {0: "negative", 1: "neutral", 2: "positive"}
                
                if isinstance(sentiment, int) and sentiment in label_map:
                    # label = label_map[sentiment]
                    label = sentiment
                elif isinstance(sentiment, str):
                    sentiment_lower = sentiment.lower()
                    if sentiment_lower in STRING_LABELS:
                        label = sentiment_lower
                    else:
                        continue
                else:
                    continue
                    
                if not text:
                    continue
                    
                yield text, label


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    download_investing_if_needed()

    seen = set()
    count = 0
    with OUT_PATH.open("w", encoding="utf-8") as w:
        for text, label in iter_texts_from_phrasebank():
            key = (text, label)
            if key not in seen:
                seen.add(key)
                w.write(json.dumps({"text": text, "label": label}) + "\n")
                count += 1

    print(f"✅ Finished! Wrote {count} samples → {OUT_PATH}")


if __name__ == "__main__":
    main()
