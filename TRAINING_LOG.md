# MLM Domain-Adaptation Training Log

**项目**: UMASS-Advanced-NLP  
**日期**: November 27, 2025  
**任务ID**: 49322623  
**用户**: xiongluo_umass_edu

---

## 📋 训练概述

### 项目功能
这是一个 **Domain-Adaptive MLM Fine-tuning** 项目，用于：
- 加载预训练的 BERT 模型（bert-base-go-emotion）
- 在生物医学文本数据上继续训练（MLM任务）
- 适应医学领域，生成医学领域的BERT模型

### 核心概念
- **不是从零Pre-training**，而是微调(Fine-tuning)
- **自监督学习**：用unlabeled数据，采用Masked Language Modeling目标
- **转移学习**：保留通用NLP能力 + 学习医学特定知识

---

## 🔧 Domain-Adaptation 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `mlm_probability` | 0.15 | 15%的token被mask，模型预测这些被mask的token |
| `learning_rate` | 5e-5 | 微调学习率（很小，防止遗忘预训练知识） |
| `warmup_steps` | 100 | 前100步线性增加学习率，稳定训练 |
| `max_steps` | 2000 | 总共训练2000步 |
| `per_device_train_batch_size` | 8 | 每批8条样本 |
| `max_seq_length` | 128 | 最大序列长度128个token |

### 为什么这些参数很重要？
- **学习率5e-5很小**：微调用的标准值，保留预训练知识
- **mlm_probability=0.15**：标准MLM设置，定义学习强度
- **较少的steps**：不需要像pre-training那样多epoch

---

## 📊 数据处理过程

### 数据生成（2025-11-27）

1. **BioASQ 生物医学文本**
   - 源文件：`data/raw/bioasq/BioASQ-train-factoid-6b-full-annotated.json`
   - 处理脚本：`src/data/clean_bioasq_for_mlm.py`
   - 输出：`data/processed/biomed_mlm.jsonl` (2,511 行, 3.7M)

2. **HuggingFace 生物医学文本**
   - 数据集：`Hmehdi515/biomedical_en-de` (英文部分)
   - 处理脚本：`src/data/clean_biomed_en_de.py`
   - 输出：`data/processed/biomed_mlm_2.jsonl` (11,145 行, 15M)

3. **总数据量**
   - 13,656 条unlabeled文本样本
   - 总大小：18.7M

### 未使用的数据
- 金融数据（`finance_mlm1.jsonl`, `finance_mlm2.jsonl`）：未生成（可选）

---

## 🚀 训练任务提交

### Slurm脚本
- 位置：`train_mlm_slurm.sh`
- 配置文件：`src/configs/mlm_bertgoemotions_biomed_only.yaml`

### 提交命令
```bash
cd /project/pi_hongyu_umass_edu/zonghai/patientedu_image/xiong_2/UMASS-Advanced-NLP
sbatch train_mlm_slurm.sh
```

### 提交时间
- 2025-11-27 (具体时间见上方任务ID)
- Job ID: **49322623**

### 任务资源
```
--time=5:00:00              # 5小时时间限制
--partition=gpu-preempt     # 抢占式GPU队列
--gres=gpu:1                # 1个GPU
--constraint=[2080ti]       # 2080Ti GPU
--mem=128G                  # 128GB内存
--cpus-per-task=8           # 8核CPU
```

### 任务状态
- 状态：PENDING (排队中)
- 预计开始：2025-12-11 03:47:40
- 预计完成：2025-12-11 08:47:40 (5小时后)

---

## 📝 查看训练进度的命令

### 查看任务状态
```bash
squeue -u $USER
```

### 查看详细信息
```bash
scontrol show job 49322623
```

### 实时查看日志
```bash
tail -f /project/pi_hongyu_umass_edu/zonghai/patientedu_image/xiong_2/UMASS-Advanced-NLP/logs/train_49322623.log
```

### 查看输出文件
```bash
ls -lh outputs/mlm_bert_goemotions_biomed/
```

---

## 🎯 预期输出文件

训练完成后，模型将保存在：
```
outputs/mlm_bert_goemotions_biomed/
├── pytorch_model.bin          # 模型权重（最重要！）
├── config.json                # 模型配置
├── vocab.txt                  # 词汇表
├── tokenizer.json             # 分词器配置
├── tokenizer_config.json      # 分词器详细配置
├── training_args.bin          # 训练参数
└── mlm_config_used.json       # 用到的YAML配置
```

### 模型大小
- 预计 ~380MB (BERT base model)

### 模型用途
这个微调后的模型可以用于：
- ✅ 医学文本分类
- ✅ 医学信息抽取
- ✅ 医学问答系统
- ✅ 医学语义相似度计算
- ✅ 其他医学NLP下游任务

---

## 🔗 相关文件

### 核心训练文件
- `src/models/pretraining_mlm.py` - 主训练脚本
- `src/configs/mlm_bertgoemotions_biomed_only.yaml` - 训练配置

### 数据处理脚本
- `src/data/clean_bioasq_for_mlm.py` - BioASQ数据处理
- `src/data/clean_biomed_en_de.py` - 生物医学多语言数据处理
- `src/data/clean_financial_for_llm.py` - 金融数据处理（未使用）
- `src/data/clean_financial_for_llm_2.py` - 金融数据处理（未使用）

### Slurm脚本
- `train_mlm_slurm.sh` - 后台训练提交脚本

---

## 📚 Background: Domain-Adaptation 原理

### 什么是Domain-Adaptation？
```
原始BERT（通用模型）
    ↓
在医学数据上继续训练
    ↓
医学领域BERT（专业模型）
```

### 为什么有效？
1. **保留通用知识**：BERT已经学到语法、基础词义等
2. **学习专业知识**：MLM任务让模型学习医学词汇和概念
3. **快速适应**：用微调代替从零训练，速度快100倍

### vs 完整Pre-training
| 方面 | Pre-training | Domain-Adaptation |
|------|-------------|------------------|
| 初始权重 | 随机 | 预训练权重 |
| 数据需求 | 数十亿token | 数百万token |
| 训练时间 | 数周 | 1-2小时 |
| 计算资源 | 数百GPU | 1-2个GPU |
| 学习率 | 1e-4 | 5e-5 (小10倍) |

---

## ⚠️ 故障排查

### 如果训练失败：
1. 检查数据文件是否存在：
   ```bash
   ls -lh data/processed/biomed_mlm*.jsonl
   ```

2. 查看日志文件：
   ```bash
   cat logs/train_49322623.err
   ```

3. 检查GPU可用性：
   ```bash
   nvidia-smi
   ```

4. 重新提交任务：
   ```bash
   sbatch train_mlm_slurm.sh
   ```

---

## 📞 快速参考

```bash
# 进入项目目录
cd /project/pi_hongyu_umass_edu/zonghai/patientedu_image/xiong_2/UMASS-Advanced-NLP

# 激活环境
module load conda/latest
conda activate mlm_training

# 查看数据
ls -lh data/processed/

# 查看任务状态
squeue -u $USER

# 查看日志
tail -f logs/train_*.log

# 查看模型
ls -lh outputs/mlm_bert_goemotions_biomed/
```

---

**最后更新**：2025-11-27  
**项目状态**：✅ 训练任务已提交到Slurm队列
