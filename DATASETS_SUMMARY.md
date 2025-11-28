# 📊 数据集总结

## 概览

项目包含 **6 个数据处理脚本**，生成 **6 个数据集**，共 **238,820 个样本**。

分为两大类：
- **原始数据 (无label)**: 53,591 samples - 用于 MLM 预训练
- **新增数据 (有label)**: 185,229 samples - 用于标签训练

---

## 📋 原始数据集 (MLM 预训练用 - 无label)

### 1️⃣ BioASQ 数据
- **脚本**: `src/data/clean_bioasq_for_llm.py`
- **数据来源**: 本地 JSON 文件 (Kaggle)
- **输出文件**: `data/processed/biomed_mlm.jsonl`
- **样本数**: 2,511
- **文件大小**: 3.69 MB
- **数据结构**: `{"text": str}`
- **说明**: 生物医学问答数据，从 BioASQ 的 snippets 中提取

### 2️⃣ 生物医学多语言数据
- **脚本**: `src/data/clean_biomed_en_de.py`
- **数据来源**: HuggingFace (`Hmehdi515/biomedical_en-de`)
- **输出文件**: `data/processed/biomed_mlm_2.jsonl`
- **样本数**: 11,145
- **文件大小**: 14.42 MB
- **数据结构**: `{"text": str}`
- **说明**: 英文生物医学文本，从多语言数据集中提取英文部分

### 3️⃣ 金融数据 (双源合并)
- **脚本**: `src/data/clean_financial_for_llm_2.py`
- **数据来源**: 
  - HuggingFace (`lukecarlate/english_finance_news`)
  - HuggingFace (`takala/financial_phrasebank`) via Parquet
- **输出文件**: `data/processed/finance_mlm1.jsonl`
- **样本数**: 20,023
- **文件大小**: 3.39 MB
- **数据结构**: `{"text": str}`
- **说明**: 金融新闻 + 金融短语银行

### 4️⃣ 金融数据 (单源)
- **脚本**: `src/data/clean_financial_for_llm.py`
- **数据来源**: HuggingFace (`lukecarlate/english_finance_news`)
- **输出文件**: `data/processed/finance_mlm2.jsonl`
- **样本数**: 19,912
- **文件大小**: 3.37 MB
- **数据结构**: `{"text": str}`
- **说明**: 金融新闻数据

**原始数据小计**: **53,591 samples**, **24.87 MB**

---

## ��️ 新增数据集 (标签训练用 - 有label)

### 5️⃣ FIQA 金融数据 (带label)
- **脚本**: `src/data/clean_fiqa_with_label.py`
- **数据来源**: HuggingFace (`llamafactory/fiqa`)
- **输出文件**: `data/processed/fiqa_labeled.jsonl`
- **样本数**: 6,486
- **文件大小**: 1.78 MB
- **数据结构**: `{"text": str, "label": str}`
- **label 说明**: 从 instruction + input (作为 text) 和 output (截断到100字作为 label)
- **数据来源不同**: ✅ 完全不同于 finance_mlm1/mlm2

### 6️⃣ MedMCQA 医学数据 (带label)
- **脚本**: `src/data/clean_pubmed_with_label.py`
- **数据来源**: HuggingFace (`medmcqa`)
- **输出文件**: `data/processed/medmcqa_labeled.jsonl`
- **样本数**: 178,743
- **文件大小**: 101.70 MB
- **数据结构**: `{"text": str, "label": str}`
- **label 分布**: 21个不同医学科目
  - 主要科目: Medicine (17,268), Surgery (16,562), Anatomy (14,276), Pathology (14,237), Pharmacology (13,245)...
- **label 说明**: question + explanation 作为 text，subject_name 作为 label
- **数据来源不同**: ✅ 完全不同于 biomed_mlm/mlm_2（后者用 BioASQ+多语言）

**新增数据小计**: **185,229 samples**, **103.48 MB**

---

## 📊 数据集对比

| 特性 | 原始数据 (无label) | 新增数据 (有label) |
|------|------------------|-----------------|
| **脚本数** | 4 | 2 |
| **样本数** | 53,591 | 185,229 |
| **文件大小** | 24.87 MB | 103.48 MB |
| **字段** | text | text + label |
| **用途** | MLM 预训练 | 标签/分类训练 |
| **最大单个数据集** | finance_mlm1 (20,023) | medmcqa (178,743) |
| **特点** | 仅包含文本内容 | 包含分类标签 |

---

## ✅ 数据质量检查

### 原始数据
- ✅ 去重处理 (基于 text 内容)
- ✅ 最小长度检查 (≥30-50字符)
- ✅ 已验证 JSONL 格式有效

### 新增数据  
- ✅ 去重处理 (基于 text 内容)
- ✅ 最小长度检查 (text ≥30-50字符, label ≥10字符)
- ✅ 已验证 JSONL 格式有效
- ✅ Label 字段完整无空值

---

## 📁 文件位置

```
data/
├── raw/
│   ├── bioasq/
│   │   └── BioASQ-train-factoid-6b-full-annotated.json
│   └── finance/
│       └── finance_news_raw.jsonl
└── processed/
    ├── biomed_mlm.jsonl          # BioASQ
    ├── biomed_mlm_2.jsonl        # 多语言生物医学
    ├── finance_mlm1.jsonl        # 金融(双源)
    ├── finance_mlm2.jsonl        # 金融(单源)
    ├── fiqa_labeled.jsonl        # 金融(有label)
    └── medmcqa_labeled.jsonl     # 医学(有label)

src/data/
├── clean_bioasq_for_mlm.py       # 原始脚本1
├── clean_biomed_en_de.py         # 原始脚本2
├── clean_financial_for_llm.py    # 原始脚本3
├── clean_financial_for_llm_2.py  # 原始脚本4
├── clean_fiqa_with_label.py      # 新增脚本1
└── clean_pubmed_with_label.py    # 新增脚本2
```

---

## 🎯 使用建议

### MLM 预训练
将原始数据 (4个脚本) 用于 Masked Language Modeling 训练：
- 结合所有 JSONL 文件或分别使用
- 数据量: 53,591 samples

### 标签/分类训练
将新增数据 (2个脚本) 用于有监督学习：
- FIQA: 金融问答 (6,486 samples)
- MedMCQA: 医学多选题 (178,743 samples)
- 总计: 185,229 samples

---

## 📝 生成日期
2025-11-27

