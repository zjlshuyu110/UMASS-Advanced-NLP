from pathlib import Path
import json
import re
from datasets import load_dataset

RAW_DIR = Path("data/raw/finance")
OUT_PATH = Path("data/processed/label_finance_mlm.jsonl")

def normalize(text: str) -> str:
    """Normalize whitespace and strip."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def download_finance_if_needed():
    """
    Download lukecarlate/english_finance_news into RAW_DIR
    if not present.
    """
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print("Downloading english_finance_news from HuggingFace...")

        RAW_DIR.mkdir(parents=True, exist_ok=True)

        # Load dataset directly
        ds = load_dataset("lukecarlate/english_finance_news", split="train")

        # Save raw JSONL for reproducibility
        raw_path = RAW_DIR / "finance_news_raw.jsonl"
        with raw_path.open("w", encoding="utf-8") as w:
            for row in ds:
                w.write(json.dumps(row) + "\n")

        print("Saved raw finance news to:", raw_path)
    else:
        print("Finance dataset already exists locally, skipping download.")


def iter_texts_from_finance():
    """
    Iterate over raw HuggingFace JSONL and yield (cleaned text, label).
    """
    for p in RAW_DIR.glob("*.jsonl"):
        print("Reading:", p)

        with p.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                txt = normalize(obj.get("newscontents", ""))
                label = obj.get("label", None)

                if len(txt) > 30 and label is not None:
                    yield txt, label


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    download_finance_if_needed()

    seen = set()
    with OUT_PATH.open("w", encoding="utf-8") as w:
        for txt, label in iter_texts_from_finance():
            key = (txt, label)
            if key not in seen:
                seen.add(key)
                w.write(json.dumps({"text": txt, "label": label}) + "\n")

    print("Finished preprocessing finance dataset →", OUT_PATH)


if __name__ == "__main__":
    main()
