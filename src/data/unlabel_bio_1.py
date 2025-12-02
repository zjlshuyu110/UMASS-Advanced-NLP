from pathlib import Path
import json
import re
import kagglehub

RAW_DIR = Path("data/raw/bio")
OUT_PATH = Path("data/processed/unlabel_bio_1.jsonl")

def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def download_bioasq_if_needed():
    """
    Download BioASQ dataset using kagglehub if it's not already present.
    """
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        print("Downloading BioASQ dataset from Kaggle...")
        path = kagglehub.dataset_download("maverickss26/bioasq-dataset")
        print("Downloaded to:", path)

        RAW_DIR.mkdir(parents=True, exist_ok=True)

        # Copy/move files into our project structure
        for file in Path(path).rglob("*"):
            if file.is_file():
                target = RAW_DIR / file.name
                target.write_bytes(file.read_bytes())
        
        print("BioASQ files copied to:", RAW_DIR)
    else:
        print("BioASQ already exists locally, skipping download.")



def iter_texts_from_bioasq():
    for p in RAW_DIR.glob("**/*.json"):
        print("Reading:", p)

        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)

        # Your BioASQ format: data -> paragraphs -> context
        if isinstance(obj, dict) and "data" in obj:
            for article in obj["data"]:
                for para in article.get("paragraphs", []):
                    txt = normalize(para.get("context", ""))
                    if len(txt) > 30:
                        yield txt

        # Fallback in case structure changes
        else:
            print("Unknown structure in:", p)



def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    download_bioasq_if_needed()

    seen = set()
    with OUT_PATH.open("w", encoding="utf-8") as w:
        for txt in iter_texts_from_bioasq():
            if txt not in seen:
                seen.add(txt)
                w.write(json.dumps({"text": txt}) + "\n")

    print("Finished preprocessing BioASQ →", OUT_PATH)


if __name__ == "__main__":
    main()
