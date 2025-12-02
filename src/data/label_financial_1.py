"""
金融数据预处理脚本 (带label)
数据来源: https://huggingface.co/datasets/llamafactory/fiqa
输出格式: {"text": str, "label": str}
FIQA结构: instruction + input (作为主要text), output (作为label)
"""

from pathlib import Path
import json
import re
from datasets import load_dataset

OUT_PATH = Path("data/processed/label_financial_1.jsonl")

def normalize(text: str) -> str:
    """Normalize whitespace and strip."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    print("Loading dataset llamafactory/fiqa...")
    dataset = load_dataset("llamafactory/fiqa")
    
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    seen = set()
    count = 0
    
    with OUT_PATH.open("w", encoding="utf-8") as f:
        # Process all splits
        for split in dataset:
            print(f"  Processing split: {split}")
            for row in dataset[split]:
                # FIQA has: instruction, input, output
                # Use instruction + input as text, output as label (truncated for reasonable label size)
                instruction = row.get("instruction", "").strip()
                input_text = row.get("input", "").strip()
                output_text = row.get("output", "").strip()
                
                # Combine instruction and input as the main text
                if input_text:
                    text = f"{instruction} {input_text}".strip()
                else:
                    text = instruction
                
                if not text or not output_text:
                    continue
                
                text = normalize(text)
                # Truncate output to reasonable label size (first 100 chars)
                label = normalize(output_text)[:100]
                
                # Minimum length check
                if len(text) < 30 or len(label) < 10:
                    continue
                
                # Deduplication
                if text in seen:
                    continue
                
                seen.add(text)
                
                # Write with label preserved
                output_row = {
                    "text": text,
                    "label": label
                }
                f.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                count += 1
    
    print(f"\n✅ Done! Saved {count} financial samples with labels to {OUT_PATH}")


if __name__ == "__main__":
    main()
