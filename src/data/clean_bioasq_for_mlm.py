from pathlib import Path
import json
import re

RAW_DIR = Path("data/raw/bioasq")
OUT_PATH = Path("data/processed/biomed_mlm.jsonl")


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def iter_texts_from_bioasq():
    """
    Iterate over all BioASQ JSON files and yield cleaned snippet texts.
    Assumes you downloaded BioASQ JSON into data/raw/bioasq/.
    """
    for p in RAW_DIR.glob("**/*.json"):
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        snippets = obj.get("snippets", [])
        for sn in snippets:
            txt = normalize(sn.get("text", ""))
            if len(txt) > 15:
                yield txt


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seen = set()

    with OUT_PATH.open("w", encoding="utf-8") as w:
        for txt in iter_texts_from_bioasq():
            if txt not in seen:
                seen.add(txt)
                w.write(json.dumps({"text": txt}) + "\n")


if __name__ == "__main__":
    main()
