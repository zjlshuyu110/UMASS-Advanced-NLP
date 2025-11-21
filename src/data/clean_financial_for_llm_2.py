from pathlib import Path
import json
import re
from datasets import load_dataset

RAW_DIR = Path("data/raw/financial")
OUT_PATH = Path("data/processed/finance_mlm1.jsonl")

def normalize(text: str) -> str:
    """Normalize whitespace and strip."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def iter_finance_news():
    print("Loading lukecarlate/english_finance_news ...")
    ds = load_dataset(
        "lukecarlate/english_finance_news",
        split="train"
    )

    for row in ds:
        txt = normalize(row.get("newscontents", ""))
        if len(txt) > 30:
            yield txt

# Exact valid parquet URL from commit that contains the Parquet file
FPB_PARQUET_URL = (
    "https://huggingface.co/datasets/takala/financial_phrasebank/"
    "resolve/4a94a2397c42fc3274b7db5632f913eb914e9e0f/"
    "sentences_50agree/financial_phrasebank-train.parquet"
)

def iter_phrasebank():
    print("Loading takala/financial_phrasebank (parquet) ...")
    fpb = load_dataset(
        "parquet",
        data_files=FPB_PARQUET_URL
    )

    for row in fpb["train"]:
        txt = normalize(row.get("sentence", ""))
        if len(txt) > 30:
            yield txt


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen = set()

    with OUT_PATH.open("w", encoding="utf-8") as w:

        # Finance News
        for txt in iter_finance_news():
            if txt not in seen:
                seen.add(txt)
                w.write(json.dumps({"text": txt}) + "\n")

        # Phrasebank
        for txt in iter_phrasebank():
            if txt not in seen:
                seen.add(txt)
                w.write(json.dumps({"text": txt}) + "\n")

    print("Finished preprocessing financial datasets →", OUT_PATH)


if __name__ == "__main__":
    main()