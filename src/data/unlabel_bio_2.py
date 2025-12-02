from datasets import load_dataset
from pathlib import Path
import json
import re

OUT_PATH = Path("data/processed/biomed_mlm_2.jsonl")

def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_english(row):
    if "text" not in row:
        return None

    t = row["text"]

    # If it's a real dict
    if isinstance(t, dict):
        if "en" in t:
            return t["en"]

    # If it's a JSON string containing en/de
    if isinstance(t, str):
        try:
            parsed = json.loads(t)
            if isinstance(parsed, dict) and "en" in parsed:
                return parsed["en"]
            else:
                return normalize(t)
        except json.JSONDecodeError:
            # plain text
            return normalize(t)

    return None

def main():
    print("Loading dataset Hmehdi515/biomedical_en-de...")
    dataset = load_dataset("Hmehdi515/biomedical_en-de")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    count = 0

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for split in dataset:
            for row in dataset[split]:
                text = extract_english(row)

                if text is None:
                    continue

                text = normalize(text)

                if len(text) < 50:
                    continue

                if text in seen:
                    continue

                seen.add(text)
                f.write(json.dumps({"text": text}) + "\n")
                count += 1

    print(f"✅ Done! Saved {count} English samples to {OUT_PATH}")

if __name__ == "__main__":
    main()
