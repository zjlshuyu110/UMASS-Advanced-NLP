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
OUT_PATH = Path("data/processed/financial_with_label_1.jsonl")


def normalize(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def download_finance_if_needed():
    """
    Download zeroshot/twitter-financial-news-sentiment dataset via HuggingFace.
    Save the raw JSONL into data/raw/finance_news.
    """
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print("Downloading financial news dataset from HuggingFace...")

        RAW_DIR.mkdir(parents=True, exist_ok=True)

        ds = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")

        raw_path = RAW_DIR / "finance_raw.jsonl"
        with raw_path.open("w", encoding="utf-8") as w:
            for row in ds:
                w.write(json.dumps(row) + "\n")

        print("Saved raw dataset →", raw_path)
    else:
        print("Finance news raw files already exist, skipping download.")


def iter_texts_from_finance():
    """
    Load the raw saved JSONL file and yield normalized tweet text + label.
    """
    for p in RAW_DIR.glob("*.jsonl"):
        print("Reading:", p)

        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except:
                    continue

                text = normalize(obj.get("text", ""))
                label = obj.get("label", None)

                if len(text) > 0:
                    yield text, label


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    download_finance_if_needed()

    seen = set()
    with OUT_PATH.open("w", encoding="utf-8") as w:
        for text, label in iter_texts_from_finance():
            key = (text, label)
            if key not in seen:
                seen.add(key)
                w.write(json.dumps({"text": text, "label": label}) + "\n")

    print("Finished preprocessing →", OUT_PATH)


if __name__ == "__main__":
    main()

