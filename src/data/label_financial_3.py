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
    Download investing_financial_news_headlines dataset from HuggingFace if not present.
    """
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print("Downloading investing_financial_news_headlines from HuggingFace...")
        ds = load_dataset("MASHXD/investing_financial_news_headlines", split="train")
        RAW_DIR.mkdir(parents=True, exist_ok=True)

        raw_path = RAW_DIR / "investing_raw.jsonl"
        with raw_path.open("w", encoding="utf-8") as w:
            for row in ds:
                w.write(json.dumps(row) + "\n")

        print("Saved raw data →", raw_path)
    else:
        print("Raw data already exists, skipping download.")


def iter_texts_from_investing():
    """
    Read raw JSONL and yield normalized text + numeric label.
    """
    for p in RAW_DIR.glob("*.jsonl"):
        print("Reading:", p)
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = normalize(obj.get("text", ""))
                sentiment = obj.get("sentiment", None)
                if not text or sentiment not in LABEL_MAP:
                    continue
                label = LABEL_MAP[sentiment]
                yield text, label


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    download_investing_if_needed()

    seen = set()
    with OUT_PATH.open("w", encoding="utf-8") as w:
        for text, label in iter_texts_from_investing():
            key = (text, label)
            if key not in seen:
                seen.add(key)
                w.write(json.dumps({"text": text, "label": label}) + "\n")

    print("Finished preprocessing →", OUT_PATH)


if __name__ == "__main__":
    main()
