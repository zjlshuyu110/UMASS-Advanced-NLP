from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
import kagglehub


PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_TRAIN = PROCESSED_DIR / "biomed_doctor_reviews_train.jsonl"
OUT_VAL   = PROCESSED_DIR / "biomed_doctor_reviews_val.jsonl"


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


def load_doctor_reviews() -> pd.DataFrame:
    """
    Download and load the doctor review dataset from Kaggle.

    Expected Kaggle columns:
      - 'reviews'  : the text
      - 'labels'   : 0 or 1  (0=negative, 1=positive)
      - sometimes 'tags'     : category (ignored here)
    """
    print("📥 Downloading doctor reviews dataset from Kaggle...")
    kaggle_path = kagglehub.dataset_download(
        "avasaralasaipavan/doctor-review-dataset-has-reviews-on-doctors"
    )
    kaggle_path = Path(kaggle_path)
    print("Downloaded to:", kaggle_path)

    csv_files = list(kaggle_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found under {kaggle_path}")
    csv_path = csv_files[0]
    print("📄 Using CSV:", csv_path)

    df = pd.read_csv(csv_path)
    print("Columns in CSV:", list(df.columns))

    # Expect at least 'reviews' and 'labels'
    if "reviews" not in df.columns:
        raise ValueError("Expected a 'reviews' column in the doctor dataset.")
    if "labels" not in df.columns:
        raise ValueError("Expected a 'labels' (0/1) column in the doctor dataset.")

    df = df.dropna(subset=["reviews", "labels"]).copy()

    # Map numeric labels -> sentiment strings
    label_map = {0: "negative", 1: "positive"}
    df["label"] = df["labels"].map(label_map)

    # Keep only valid labels
    df = df[df["label"].isin(["negative", "positive"])]

    df["text"] = df["reviews"].astype(str)
    df = df[["text", "label"]]

    print(f"Doctor reviews: {len(df)} rows after cleaning.")
    return df


def main(test_size: float = 0.1, random_state: int = 42):
    df = load_doctor_reviews()

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    write_jsonl(train_df, OUT_TRAIN)
    write_jsonl(val_df, OUT_VAL)
    print("🎉 Finished preparing doctor review sentiment data for SFT.")


if __name__ == "__main__":
    main()
