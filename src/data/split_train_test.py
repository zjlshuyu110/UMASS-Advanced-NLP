"""
将标注数据分割为训练集和测试集，并创建平衡的混合数据集
- Bio数据: 80% train / 20% test
- Finance数据: 80% train / 20% test  
- Mixed平衡数据: 从Bio和Finance各取相同数量，保证50/50平衡
"""

import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)

# 文件路径
data_dir = Path("data/processed")
bio_files = ["label_bio_drug_reviews.jsonl", "label_bio_2.jsonl", "label_bio_4.jsonl"]
finance_files = ["label_financial_2.jsonl", "label_financial_3.jsonl"]

# 读取所有数据
print("=== 读取数据 ===")
bio_data = []
finance_data = []

for f in bio_files:
    path = data_dir / f
    if path.exists():
        with open(path) as file:
            data = [json.loads(line) for line in file]
            bio_data.extend(data)
            print(f"{f}: {len(data)} samples")

for f in finance_files:
    path = data_dir / f
    if path.exists():
        with open(path) as file:
            data = [json.loads(line) for line in file]
            finance_data.extend(data)
            print(f"{f}: {len(data)} samples")

print(f"\nBio总数: {len(bio_data)}")
print(f"Finance总数: {len(finance_data)}")

# 打乱数据
random.shuffle(bio_data)
random.shuffle(finance_data)

# 分割train/test (80/20)
bio_train_size = int(len(bio_data) * 0.8)
finance_train_size = int(len(finance_data) * 0.8)

bio_train = bio_data[:bio_train_size]
bio_test = bio_data[bio_train_size:]

finance_train = finance_data[:finance_train_size]
finance_test = finance_data[finance_train_size:]

print("\n=== Train/Test分割 ===")
print(f"Bio train: {len(bio_train)}, test: {len(bio_test)}")
print(f"Finance train: {len(finance_train)}, test: {len(finance_test)}")

# 创建平衡的混合数据集
# Bio总共7,050，Finance总共14,377
# 为了平衡，从两边各取7,050个样本
balance_size_total = len(bio_data)  # 7,050 (较小的那个)

# 从finance中随机抽取7,050个
finance_sampled = random.sample(finance_data, balance_size_total)

# 合并并分割train/test (80/20)
all_mixed = bio_data + finance_sampled
random.shuffle(all_mixed)

mixed_train_size = int(len(all_mixed) * 0.8)
mixed_train = all_mixed[:mixed_train_size]
mixed_test = all_mixed[mixed_train_size:]

print(f"\nMixed balanced (total): {len(all_mixed)} ({len(bio_data)} bio + {len(finance_sampled)} finance)")
print(f"Mixed balanced train: {len(mixed_train)}")
print(f"Mixed balanced test: {len(mixed_test)}")

# 保存文件
def save_jsonl(data, filepath):
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
def print_label_dist(data, name):
    labels = Counter([d['label'] for d in data])
    print(f"{name}: neg={labels['negative']}, neu={labels['neutral']}, pos={labels['positive']}")

print("\n=== 保存文件 ===")

# Bio数据
save_jsonl(bio_train, data_dir / "bio_train.jsonl")
save_jsonl(bio_test, data_dir / "bio_test.jsonl")
print_label_dist(bio_train, "Bio train")
print_label_dist(bio_test, "Bio test")

# Finance数据
save_jsonl(finance_train, data_dir / "finance_train.jsonl")
save_jsonl(finance_test, data_dir / "finance_test.jsonl")
print_label_dist(finance_train, "Finance train")
print_label_dist(finance_test, "Finance test")

# Mixed平衡数据
save_jsonl(mixed_train, data_dir / "mixed_balanced_train.jsonl")
save_jsonl(mixed_test, data_dir / "mixed_balanced_test.jsonl")
print_label_dist(mixed_train, "Mixed train")
print_label_dist(mixed_test, "Mixed test")

print("\n✅ 数据分割完成！")
print(f"\n生成的文件:")
print(f"  - bio_train.jsonl ({len(bio_train)} samples)")
print(f"  - bio_test.jsonl ({len(bio_test)} samples)")
print(f"  - finance_train.jsonl ({len(finance_train)} samples)")
print(f"  - finance_test.jsonl ({len(finance_test)} samples)")
print(f"  - mixed_balanced_train.jsonl ({len(mixed_train)} samples)")
print(f"  - mixed_balanced_test.jsonl ({len(mixed_test)} samples)")
