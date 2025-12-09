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
    """Map 1–5 ratings to negative/neutral/positive sentiment labels."""
    try:
        r = int(rating)
    except (TypeError, ValueError):
        return None

    if r in (1, 2):
        return "negative"
    if r == 3:
        return "neutral"
    if r in (4, 5):
        return "positive"
    return None


def download_if_needed():
    """
    Download hospital-reviews-dataset using kagglehub if it's not already present.
    """
    # Check if we already have hospital review CSV files
    hospital_csvs = list(RAW_DIR.glob("*hospital*.csv")) if RAW_DIR.exists() else []
    
    if not hospital_csvs:
        print("Downloading hospital-reviews-dataset from Kaggle...")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        
        path = kagglehub.dataset_download("junaid6731/hospital-reviews-dataset")
        print("Downloaded to:", path)

        # Copy CSV files into our RAW_DIR
        num_copied = 0
        for file in Path(path).rglob("*.csv"):
            target = RAW_DIR / file.name
            target.write_bytes(file.read_bytes())
            num_copied += 1

        print(f"Copied {num_copied} CSV file(s) to {RAW_DIR}")
    else:
        print(f"Hospital CSV files already exist in {RAW_DIR}")


def iter_reviews():
    """Yield (text, label) pairs from hospital CSV files in RAW_DIR."""
    csv_files = list(RAW_DIR.glob("*hospital*.csv"))
    if not csv_files:
        print(f"Warning: No hospital CSV files found in {RAW_DIR}")
        return
        
    for p in csv_files:
        print("Reading:", p)
        try:
            with p.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                # Print first row to debug column names
                first_row = next(reader, None)
                if first_row:
                    print(f"Columns found: {list(first_row.keys())}")
                    # Try different possible column names
                    text_col = None
                    rating_col = None
                    
                    for col in first_row.keys():
                        if col.lower() in ['feedback', 'review', 'text', 'comment']:
                            text_col = col
                        if col.lower() in ['ratings', 'rating', 'score']:
                            rating_col = col
                    
                    if not text_col or not rating_col:
                        print(f"Warning: Could not find text/rating columns in {p}")
                        continue
                    
                    # Process first row
                    text = normalize(first_row.get(text_col, ""))
                    rating = first_row.get(rating_col)
                    if text and len(text) >= 5:
                        label = rating_to_label(rating)
                        if label is not None:
                            yield text, label
                    
                    # Process remaining rows
                    for row in reader:
                        text = normalize(row.get(text_col, ""))
                        rating = row.get(rating_col)

                        if not text or len(text) < 5:
                            continue

                        label = rating_to_label(rating)
                        if label is None:
                            continue

                        yield text, label
        except Exception as e:
            print(f"Error reading {p}: {e}")
            continue


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


