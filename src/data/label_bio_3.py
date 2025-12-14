from pathlib import Path
import json
import re
import pandas as pd
from typing import Optional


# --------- Paths ---------
RAW_CSV = Path(
    "/Users/user/Downloads/DrugRev A comprehensive customers reviews on drugs purchasing and satisfaction/DrugReviews.csv"
)  # change if needed

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = PROCESSED_DIR / "label_bio_3_strict.jsonl"

# Target total size (after cleaning)
MAX_SAMPLES = 15000

# Filtering knobs
MIN_TOKENS = 20
MAX_TOKENS = 280          # 250-300 is a good range
DROP_CONTRAST = False     # set True to drop mixed-sentiment "but/however" style reviews

# If your dataset uses Satisfaction on a 1-5 scale, set this True
SCALE_1_TO_5 = False      # set True only if the ratings are 1..5


CONTRAST_RE = re.compile(r"\b(but|however|although|though|yet|nevertheless)\b", re.IGNORECASE)


def download_drug_reviews_if_needed():
    """
    Optional: download a Kaggle dataset if RAW_CSV doesn't exist.
    NOTE: you might need to change dataset id to match your exact Kaggle source.
    """
    if RAW_CSV.exists():
        print(f"✅ Found local CSV: {RAW_CSV}")
        return

    print("📥 RAW_CSV not found. Downloading drug reviews dataset from Kaggle via kagglehub...")
    import kagglehub

    # If this isn't your dataset, change the Kaggle id here:
    path = kagglehub.dataset_download("rohanharode07/webmd-drug-reviews-dataset")
    print("Downloaded to:", path)

    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)

    csvs = list(Path(path).rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError("No CSV found inside downloaded Kaggle dataset folder.")

    # Copy the first CSV found into RAW_CSV path
    RAW_CSV.write_bytes(csvs[0].read_bytes())
    print(f"✅ Copied {csvs[0].name} → {RAW_CSV}")


def map_rating_to_label(r: float) -> Optional[str]:
    """
    Strong-label mapping to reduce noise:
      negative: 1-3
      neutral:  5-6
      positive: 8-10
    Drop ambiguous ratings 4 and 7.

    Returns:
      "0" for negative, "1" for neutral, "2" for positive, or None to drop.
    """
    if SCALE_1_TO_5:
        # Convert 1..5 into something comparable:
        # 1-2 -> neg, 3 -> neutral, 4-5 -> pos
        if r in (1, 2):
            return "0"
        if r == 3:
            return "1"
        if r in (4, 5):
            return "2"
        return None

    # 1..10 scale
    if 1 <= r <= 3:
        return "0"
    if 5 <= r <= 6:
        return "1"
    if 8 <= r <= 10:
        return "2"
    return None  # drop 4 and 7 (and anything weird)


def count_tokens(text: str) -> int:
    # simple whitespace token count (good enough for filtering)
    return len(text.split())


def write_jsonl(df: pd.DataFrame, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({"text": row["text"], "label": row["label"]}) + "\n")
    print(f"✅ Wrote {len(df)} rows → {path}")


def stratified_subsample(df: pd.DataFrame, max_total: int, random_state: int = 42) -> pd.DataFrame:
    if len(df) <= max_total:
        print(f"Dataset has {len(df)} rows (<= {max_total}), no subsampling needed.")
        return df

    print(f"🔻 Subsampling from {len(df)} → {max_total} rows (stratified by label)")
    labels = sorted(df["label"].unique())
    per = max_total // len(labels)

    parts = []
    for lab in labels:
        g = df[df["label"] == lab]
        k = min(len(g), per)
        parts.append(g.sample(n=k, random_state=random_state))
        print(f"  label {lab}: {k} sampled (from {len(g)})")

    out = pd.concat(parts, ignore_index=True)

    # top-up if needed
    remaining = max_total - len(out)
    if remaining > 0:
        pool = df.drop(out.index, errors="ignore")
        if len(pool) > 0:
            extra = pool.sample(n=min(remaining, len(pool)), random_state=random_state)
            out = pd.concat([out, extra], ignore_index=True)

    print("Label distribution (after subsampling):\n", out["label"].value_counts())
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def main(random_state: int = 42):
    download_drug_reviews_if_needed()

    print(f"📄 Loading: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    print("Columns:", list(df.columns))

    # Identify text + rating columns
    if "Reviews" not in df.columns:
        raise ValueError("Expected a 'Reviews' column.")
    rating_col = "Rating"
    if rating_col not in df.columns:
        # fallback for datasets that call it Satisfaction
        if "Satisfaction" in df.columns:
            rating_col = "Satisfaction"
        else:
            raise ValueError("Expected a 'Rating' (or 'Satisfaction') column.")

    df = df[["Reviews", rating_col]].copy()
    df = df.dropna(subset=["Reviews", rating_col])

    df["rating"] = pd.to_numeric(df[rating_col], errors="coerce")
    df = df.dropna(subset=["rating"])

    # normalize text
    df["text"] = df["Reviews"].astype(str).str.strip()

    # map rating -> label, dropping ambiguous
    df["label"] = df["rating"].apply(map_rating_to_label)
    before = len(df)
    df = df[df["label"].notna()].copy()
    print(f"🧹 Dropped {before - len(df)} rows due to ambiguous ratings.")

    # length filtering
    df["tok_len"] = df["text"].apply(count_tokens)
    before = len(df)
    df = df[(df["tok_len"] >= MIN_TOKENS) & (df["tok_len"] <= MAX_TOKENS)].copy()
    print(f"🧹 Dropped {before - len(df)} rows due to token-length filter [{MIN_TOKENS}, {MAX_TOKENS}].")

    # optional contrast filtering
    if DROP_CONTRAST:
        before = len(df)
        df = df[~df["text"].str.contains(CONTRAST_RE, regex=True)].copy()
        print(f"🧹 Dropped {before - len(df)} rows due to contrast-word filter.")

    # drop duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"🧹 Dropped {before - len(df)} duplicate texts.")

    print("✅ Label distribution (cleaned):\n", df["label"].value_counts())

    # stratified subsample
    df = df[["text", "label"]].reset_index(drop=True)
    df_sub = stratified_subsample(df, MAX_SAMPLES, random_state=random_state)

    write_jsonl(df_sub, OUT_PATH)
    print(f"🎉 Done: {len(df_sub)} samples saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
