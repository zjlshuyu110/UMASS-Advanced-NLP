from collections import Counter
import json

def check_label_counts(path):
    counts = Counter()
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            counts[obj["label"]] += 1
    print(path, counts)

check_label_counts("data/processed/bio_combined.jsonl")
check_label_counts("data/processed/finance_combined.jsonl")
