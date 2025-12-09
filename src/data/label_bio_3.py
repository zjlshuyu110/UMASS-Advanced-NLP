from pathlib import Path
import json

import pandas as pd


# --------- Paths ---------
RAW_CSV = Path(
    "/Users/user/Downloads/DrugRev A comprehensive customers reviews on drugs purchasing and satisfaction/DrugReviews.csv"
)  # <-- change if needed

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_ALL = PROCESSED_DIR / "biomed_drug_reviews.jsonl"

MAX_SAMPLES = 9000  # target total size


def rating_to_label(rating: float) -> str:
    """
    Map 1-10 rating to sentiment label:
      1-4  -> negative
      5-6  -> neutral
      7-10 -> positive
    """
    if rating <= 4:
        return "negative"
    elif rating <= 6:
        return "neutral"
    else:
        return "positive"


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


def load_and_clean_drug_reviews() -> pd.DataFrame:
    """
    Load the raw drug reviews CSV and normalize to columns: text, label.

    Expected columns (at least):
      - Reviews : the free-text review
      - Rating  : numeric rating 1–10
    """
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {RAW_CSV}. Place the drug review CSV there."
        )

    print(f"📄 Loading drug reviews from {RAW_CSV} ...")
    df = pd.read_csv(RAW_CSV)

    print("Columns in CSV:", list(df.columns))

    # Make sure the expected columns exist
    if "Reviews" not in df.columns:
        raise ValueError("Expected a 'Reviews' column in the drug reviews dataset.")
    if "Rating" not in df.columns:
        raise ValueError("Expected a 'Rating' column in the drug reviews dataset.")

    # Keep only relevant columns for now
    df = df[["Reviews", "Rating"]].copy()

    # Drop rows with missing reviews or ratings
    df = df.dropna(subset=["Reviews", "Rating"])

    # Ensure Rating is numeric
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=["Rating"])

    # Map rating -> sentiment label
    df["label"] = df["Rating"].apply(rating_to_label)

    # Normalize text
    df["text"] = df["Reviews"].astype(str).str.strip()

    # Keep only what we need
    df = df[["text", "label"]]

    # Drop exact duplicate texts
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    after = len(df)
    print(f"🧹 Dropped {before - after} duplicate reviews. Remaining: {after}")

    print("Label distribution (full dataset):\n", df["label"].value_counts())

    return df


def stratified_subsample(
    df: pd.DataFrame, max_total: int, random_state: int = 42
) -> pd.DataFrame:
    """
    Subsample the dataset to at most max_total rows, stratified by label.
    If dataset is already smaller, returns as-is.
    """
    n = len(df)
    if n <= max_total:
        print(f"Dataset has {n} rows (<= {max_total}), no subsampling needed.")
        return df

    print(f"Subsampling from {n} → {max_total} rows (stratified by label)...")

    num_labels = df["label"].nunique()
    base_per_label = max_total // num_labels

    # First pass: sample up to base_per_label per label
    parts = []
    for label, group in df.groupby("label"):
        k = min(len(group), base_per_label)
        sampled = group.sample(n=k, random_state=random_state)
        parts.append(sampled)
        print(f"  Label '{label}': sampled {k} rows (group had {len(group)})")

    subsampled = pd.concat(parts, ignore_index=True)

    # If we have fewer than max_total because some labels were small,
    # we can sample additional rows from the remaining pool.
    remaining = max_total - len(subsampled)
    if remaining > 0:
        print(f"  Sampling additional {remaining} rows from remaining pool.")
        df_remaining = df.drop(subsampled.index)
        if len(df_remaining) > 0:
            extra = df_remaining.sample(
                n=min(remaining, len(df_remaining)),
                random_state=random_state,
            )
            subsampled = pd.concat([subsampled, extra], ignore_index=True)

    print("Label distribution (after subsampling):\n", subsampled["label"].value_counts())
    print(f"Final subsampled size: {len(subsampled)}")
    return subsampled


def main(random_state: int = 42):
    df = load_and_clean_drug_reviews()

    # Subsample to max_total rows (stratified)
    df_sub = stratified_subsample(df, max_total=MAX_SAMPLES, random_state=random_state)

    # Write everything into ONE combined JSONL
    write_jsonl(df_sub, OUT_ALL)

    print("🎉 Finished preparing *combined* drug review sentiment data for SFT.")


if __name__ == "__main__":
    main()
