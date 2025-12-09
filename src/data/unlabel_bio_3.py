from pathlib import Path
import json
import re
from datasets import load_dataset

RAW_DIR = Path("data/raw/bio")
OUT_PATH = Path("data/processed/unlabel_bio_3.jsonl")

def normalize(text: str) -> str:
    """Normalize whitespace and strip."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def download_medical_if_needed():
    """ 
    Download medical_text into RAW_DIR
    if not present.
    """
    raw_path = RAW_DIR / "medical_text_raw.jsonl"
    
    if not raw_path.exists():
        print("Downloading medical_text from HuggingFace...")

        RAW_DIR.mkdir(parents=True, exist_ok=True)

        # Load dataset directly
        ds = load_dataset("123rc/medical_text", split="train")

        # Save raw JSONL for reproducibility
        with raw_path.open("w", encoding="utf-8") as w:
            for row in ds:
                w.write(json.dumps(row) + "\n")

        print("Saved raw medical text to:", raw_path)
    else:
        print("Medical text dataset already exists locally, skipping download.")


def iter_texts_from_medical():
    """
    Iterate over raw HuggingFace JSONL and yield cleaned text.
    """
    raw_path = RAW_DIR / "medical_text_raw.jsonl"
    
    if raw_path.exists():
        print("Reading:", raw_path)
        with raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                txt = normalize(obj.get("medical_abstract", ""))

                if len(txt) > 30:
                    yield txt


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    download_medical_if_needed()

    seen = set()
    with OUT_PATH.open("w", encoding="utf-8") as w:
        for txt in iter_texts_from_medical():
            if txt not in seen:
                seen.add(txt)
                w.write(json.dumps({"text": txt}) + "\n")

    print(f"✅ Done! Saved {len(seen)} medical text samples to {OUT_PATH}")


if __name__ == "__main__":
    main()