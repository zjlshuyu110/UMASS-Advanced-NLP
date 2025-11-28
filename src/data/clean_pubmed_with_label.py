"""
生物医学数据预处理脚本 (带label)
数据来源: https://huggingface.co/datasets/medmcqa
输出格式: {"text": str, "label": str}
MedMCQA: 医学多选题数据集，包含question和标签（subject_name作为label）
"""

from pathlib import Path
import json
import re
from datasets import load_dataset

OUT_PATH = Path("data/processed/medmcqa_labeled.jsonl")

def normalize(text: str) -> str:
    """Normalize whitespace and strip."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    print("Loading dataset medmcqa...")
    dataset = load_dataset("medmcqa")
    
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    seen = set()
    count = 0
    
    with OUT_PATH.open("w", encoding="utf-8") as f:
        # Process all splits
        for split in dataset:
            print(f"  Processing split: {split}")
            for row in dataset[split]:
                # MedMCQA结构: question + options + explanation
                # 使用question + explanation作为text，subject_name作为label
                question = row.get("question", "")
                explanation = row.get("exp", "")
                subject = row.get("subject_name", "Medical")
                
                # Handle None values
                if question:
                    question = question.strip()
                if explanation:
                    explanation = explanation.strip()
                if subject:
                    subject = subject.strip()
                else:
                    subject = "Medical"
                
                # Combine question and explanation
                if question and explanation:
                    text = f"{question} {explanation}"
                elif question:
                    text = question
                else:
                    continue
                
                text = normalize(text)
                
                # Minimum length check
                if len(text) < 50:
                    continue
                
                # Deduplication
                if text in seen:
                    continue
                
                seen.add(text)
                
                # Write with label
                output_row = {
                    "text": text,
                    "label": subject
                }
                f.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                count += 1
    
    print(f"\n✅ Done! Saved {count} biomedical samples with labels to {OUT_PATH}")


if __name__ == "__main__":
    main()
