from pathlib import Path
import json
import re
import csv
import kagglehub

RAW_DIR = Path("data/raw/bio")
OUT_PATH = Path("data/processed/label_bio_4.jsonl")


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text))
    return text.strip()


def rating_to_label(rating):
    """Map 1–5 ratings to 0/1/2 sentiment labels."""
    try:
        r = int(rating)
    except (TypeError, ValueError):
        return None

    if r in (1, 2):
        return 0  # negative
    if r == 3:
        return 1  # neutral
    if r in (4, 5):
        return 2  # positive
    return None


def download_if_needed():
    """
    Download hospital-reviews-dataset using kagglehub if it's not already present.
    """
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print("Downloading hospital-reviews-dataset from Kaggle...")
        path = kagglehub.dataset_download("junaid6731/hospital-reviews-dataset")
        print("Downloaded to:", path)

        RAW_DIR.mkdir(parents=True, exist_ok=True)

        # Copy CSV files into our RAW_DIR
        num_copied = 0
        for file in Path(path).rglob("*.csv"):
            target = RAW_DIR / file.name
            target.write_bytes(file.read_bytes())
            num_copied += 1

        print(f"Copied {num_copied} CSV file(s) to {RAW_DIR}")
    else:
        print("RAW_DIR already has files, skipping download.")


def iter_reviews():
    """Yield (text, label) pairs from all CSV files in RAW_DIR."""
    for p in RAW_DIR.glob("*.csv"):
        print("Reading:", p)
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = normalize(row.get("Feedback", ""))
                rating = row.get("Ratings")

                if not text or len(text) < 5:
                    continue

                label = rating_to_label(rating)
                if label is None:
                    continue

                yield text, label


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    download_if_needed()

    seen = set()
    n = 0
    with OUT_PATH.open("w", encoding="utf-8") as w:
        for text, label in iter_reviews():
            if text in seen:
                continue
            seen.add(text)
            w.write(json.dumps({"text": text, "label": label}) + "\n")
            n += 1

    print(f"Wrote {n} examples → {OUT_PATH}")


if __name__ == "__main__":
    main()


