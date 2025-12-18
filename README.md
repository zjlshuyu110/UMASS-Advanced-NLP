# UMASS-Advanced-NLP: Multi-Stage Sentiment Classification

## 📊 Project Overview

This project implements a comprehensive multi-stage training pipeline for sentiment classification, exploring the impact of domain-specific Masked Language Modeling (MLM) pretraining on downstream task performance.

### 🎯 Research Question
How does domain-specific MLM pretraining affect the cross-domain generalization ability of sentiment classification models?

### 🔄 Training Pipeline

**Stage 1: MLM Pretraining (3 models)**
- `mlm_bert_goemotions_biomed/` - Biomedical domain MLM
- `mlm_bert_goemotions_biomed_finance/` - Mixed biomedical + finance domain MLM
- `mlm_bert_goemotions_finance/` - Finance domain MLM

**Stage 2: Supervised Fine-tuning (7 models)**
- **SFT-1**: Baseline (bert-base-go-emotion, no MLM)
- **SFT-2**: Biomed MLM + Full finetune + Bio data
- **SFT-3**: Finance MLM + Full finetune + Finance data
- **SFT-4**: Mixed MLM + Full finetune + Mixed data
- **SFT-5**: Biomed MLM + LoRA adapter + Bio data
- **SFT-6**: Finance MLM + LoRA adapter + Finance data
- **SFT-7**: Mixed MLM + LoRA adapter + Mixed data

## 📁 Project Structure

```
UMASS-Advanced-NLP/
├── configs/                    # YAML configuration files
├── data/                       # Training and test datasets
│   ├── label_mixed_3_*.jsonl   # Main mixed dataset (Bio + Finance)
│   └── processed/              # Domain-specific processed datasets
├── models/                     # Trained model checkpoints
│   ├── mlm_*/                  # MLM pretrained models
│   └── sft_*/                  # Fine-tuned sentiment models
├── notebooks/                  # Jupyter notebooks (SFT2-SFT7)
├── results/                    # Evaluation results and analysis
│   ├── evaluation_summary.json # All model performance metrics
│   ├── error_analysis_sft*.json # Error analysis for each model
│   └── zero_shot_game_results.txt # Cross-domain evaluation
├── scripts/                    # Training and evaluation scripts
│   ├── train_sft1_*.sh         # SFT-1 training scripts
│   └── zero_shot_eval_game_reviews.py # Zero-shot evaluation
└── src/                        # Source code
    ├── models/                 # Python training/evaluation scripts
    └── data/                   # Data processing utilities
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Training SFT-1 Models
```bash
# From project root
cd scripts/

# Train baseline model
./train_sft1_baseline.sh

# Train enhanced model
./train_sft1_enhanced.sh
```

### Evaluation
```bash
# Run all model evaluations
python src/models/evaluate_all_models.py

# Zero-shot evaluation on Game Reviews
python scripts/zero_shot_eval_game_reviews.py
```

### Error Analysis
```bash
# Generate error analysis for SFT-1 baseline
python src/models/generate_error_analysis.py \
  --model_dir ../models/sft_1_baseline_new \
  --data_file ../data/label_mixed_3_test.jsonl \
  --output_file ../results/error_analysis_sft1_baseline_test.json \
  --sample_size 100
```

## 📈 Results Summary

| Model | Test Accuracy | Key Features |
|-------|---------------|--------------|
| SFT-1 Baseline | ~47% | No MLM, bert-base |
| SFT-1 Enhanced | ~65% | New data, bert-large |
| SFT-2-7 | Varies | Domain-specific MLM |

*Detailed results in `results/evaluation_summary.json`*

## 🔧 Configuration

All training configurations are in `configs/` directory:
- `mlm_*.yaml` - MLM pretraining configs
- `sft_*.yaml` - Supervised fine-tuning configs

## ⚙️ Configurations Summary

Below is a quick reference table of key hyperparameters and outputs from `configs/*.yaml`.

| Config File | Model | Train Files (short) | Batch | LR | Steps/Epochs | LoRA | Output Dir |
|---|---|---:|---:|---:|---:|---:|---|
| `mlm_bertgoemotions.yaml` | bhadresh-savani/bert-base-go-emotion | biomed_mlm_*, finance_mlm* | 8 | 5e-5 | max_steps=2000 | n/a | `../models/mlm_bert_goemotions` |
| `mlm_bertgoemotions_biomed_finance.yaml` | bhadresh-savani/bert-base-go-emotion | unlabel_bio_*, unlabel_financial_* | 8 | 5e-5 | max_steps=2000 | n/a | `../models/mlm_bert_goemotions_biomed_finance` |
| `mlm_bertgoemotions_biomed_only.yaml` | bhadresh-savani/bert-base-go-emotion | unlabel_bio_* | 8 | 5e-5 | max_steps=2000 | n/a | `../models/mlm_bert_goemotions_biomed` |
| `mlm_bertgoemotions_finance_only.yaml` | bhadresh-savani/bert-base-go-emotion | unlabel_financial_* | 8 | 5e-5 | max_steps=2000 | n/a | `../models/mlm_bert_goemotions_finance` |
| `mlm_test_small.yaml` | bhadresh-savani/bert-base-go-emotion | biomed_mlm* (small) | 8 | 5e-5 | max_steps=10 | n/a | `../models/mlm_test_small` |
| `sft_1_baseline_goemotion.yaml` | bhadresh-savani/bert-base-go-emotion | mixed_balanced_train | 8 | 5e-5 | epochs=3 | false | `../models/sft_1_baseline_goemotion` |
| `sft_2_biomed_mlm_no_adapter.yaml` | ../models/mlm_bert_goemotions_biomed | bio_train | 8 | 5e-5 | epochs=3 | false | `../models/sft_2_biomed_mlm_no_adapter` |
| `sft_3_finance_mlm_no_adapter.yaml` | ../models/mlm_bert_goemotions_finance | finance_train | 8 | 5e-5 | epochs=3 | false | `../models/sft_3_finance_mlm_no_adapter` |
| `sft_4_mixed_mlm_no_adapter.yaml` | ../models/mlm_bert_goemotions_biomed_finance | mixed_balanced_train | 8 | 5e-5 | epochs=3 | false | `../models/sft_4_mixed_mlm_no_adapter` |
| `sft_5_biomed_mlm_with_adapter.yaml` | ../models/mlm_bert_goemotions_biomed | bio_train | 8 | 5e-5 | epochs=3 | true (r=8,a=16,d=0.1) | `../models/sft_5_biomed_mlm_with_adapter` |
| `sft_6_finance_mlm_with_adapter.yaml` | ../models/mlm_bert_goemotions_finance | finance_train | 8 | 5e-5 | epochs=3 | true (r=8,a=16,d=0.1) | `../models/sft_6_finance_mlm_with_adapter` |
| `sft_7_mixed_mlm_with_adapter.yaml` | ../models/mlm_bert_goemotions_biomed_finance | mixed_balanced_train | 8 | 5e-5 | epochs=3 | true (r=8,a=16,d=0.1) | `../models/sft_7_mixed_mlm_with_adapter` |


## 📝 Key Findings

1. **Domain-specific MLM improves performance** on respective domains
2. **Mixed-domain MLM** provides best generalization
3. **LoRA adapters** reduce training time with minimal performance loss
4. **Cross-domain zero-shot** performance ~40-50% on Game Reviews dataset
 
## 🔧 Installation

1. Create and activate a Python environment (recommended `conda`):

```bash
conda create -n nlpenv python=3.10 -y
conda activate nlpenv
pip install -r requirements.txt
```

2. Optional: if you use `conda` module on cluster, load it and activate the appropriate environment before running scripts.

## ⬇️ Data download & preparation

1. Place raw JSONL datasets in `data/raw/` or use the repository's provided `data/label_mixed_3_*.jsonl` files. If you need external datasets (e.g., Game Reviews for zero-shot), download as follows:

```bash
# from project root
python - <<'PY'
from datasets import load_dataset
ds = load_dataset('auphong2707/game-reviews-sentiment')
ds['test'].to_json('data/processed/game_reviews_test.jsonl')
PY
```

2. Processed domain splits expected under `data/processed/` (the configs refer to these paths). If you need to generate processed splits from `label_mixed_3_*.jsonl`, use `src/data` utility scripts or run your own preprocessing to produce files with `text` and `label` fields.

## 📂 Directory & Data Structure (what to include in the README)

Project root (important folders shown):

```
data/                      # Raw and processed JSONL datasets
  label_mixed_3_train.jsonl
  label_mixed_3_val.jsonl
  label_mixed_3_test.jsonl
  processed/                # domain-specific splits used by configs

configs/                   # YAML experiment configs (mlm_* and sft_*)
models/                    # Trained checkpoints (mlm_* and sft_*)
scripts/                   # Convenience run scripts (train wrappers, zero-shot)
src/                       # Core implementations (mlm, sft, eval, error-analysis)
results/                   # evaluation summaries & error analysis files
notebooks/                 # exploratory/colab notebooks (SFT2..SFT7)
```

Files used by training code:
- Training expects JSONL files with at least `text` and `label` fields. Labels can be strings or ints.
- `configs/*.yaml` point to `../data/processed/` and `../models/` after reorganization.

## ▶️ How to run (examples)

Run SFT-1 baseline (wrapper script, uses configured paths):

```bash
cd scripts
./train_sft1_baseline.sh
```

Run SFT-1 enhanced (BERT-large):

```bash
cd scripts
./train_sft1_enhanced.sh
```

Run evaluation for a single model (example):

```bash
python src/models/evaluate_model.py \
  --model_dir ../models/sft_1_baseline_new \
  --test_file ../data/label_mixed_3_test.jsonl \
  --label_map ../data/processed/label_map.json \
  --output ../results/sft1_baseline_eval.json
```

Generate error analysis (sample of 100 errors):

```bash
python src/models/generate_error_analysis.py \
  --model_dir ../models/sft_1_baseline_new \
  --data_file ../data/label_mixed_3_val.jsonl \
  --output_file ../results/error_analysis_sft1_baseline_val.json \
  --sample_size 100 --save_metrics
```

Run zero-shot evaluation (game reviews):

```bash
python scripts/zero_shot_eval_game_reviews.py
```

## 🧭 How training is organized (short description for README)

- MLM pretraining: run `src/models/pretraining_mlm.py` with a `configs/mlm_*.yaml` file that lists `train_files` and `output_dir`. The script concatenates datasets, tokenizes, groups into chunks, and trains `AutoModelForMaskedLM` via `Trainer`.
- Supervised fine-tuning: run `src/models/finetune_sft1_enhanced.py` (or use the wrapper in `scripts/`) to fine-tune `AutoModelForSequenceClassification`. The script builds a `label_map.json` inside the output directory and saves `test_metrics.json` if `--test_file` is provided.
- LoRA adapters: enabled by `use_lora: true` in YAML; `evaluate_model.py` will detect and correctly load LoRA models.

## ✅ Reproducibility & tips

- Use the provided `configs/*.yaml` to reproduce each experiment. After reorganization these configs use relative paths (`../data/processed`, `../models`).
- Scripts in `scripts/` are simple wrappers that call `src/models/*` scripts with prefilled arguments.
- If GPU/cluster-specific modules are needed, load them before running (e.g., `module load conda/latest` then `conda activate discharge`).
- If you encounter CUDA compatibility issues, use a newer GPU node or adjust PyTorch/CUDA builds.

---

## 🤝 Contributing
 
This project explores multi-stage training for sentiment classification with domain adaptation techniques.