from datasets import load_dataset
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

OUT_DIR = Path("data/processed")


def main():
    ds = load_dataset("Shekswess/ai-healthcare-biomedical-sentiment")
    df = pd.DataFrame(ds["train"])

    # Inspect columns once manually later if needed
    # print(df.columns)

    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else df.columns[1]

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["text"] = df["text"].astype(str).str.strip()
    df["domain"] = "biomed"

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_json(OUT_DIR / "biomed_sentiment_train.jsonl", orient="records", lines=True)
    val_df.to_json(OUT_DIR / "biomed_sentiment_val.jsonl", orient="records", lines=True)
    test_df.to_json(OUT_DIR / "biomed_sentiment_test.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
