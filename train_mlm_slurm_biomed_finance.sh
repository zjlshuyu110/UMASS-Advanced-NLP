#!/bin/bash
#SBATCH --job-name=mlm_biomed_finance
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --time=5:00:00
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --priority=HIGH

# 加载conda
module load conda/latest

# 激活环境
conda activate mlm_training

# 进入项目目录
cd /project/pi_hongyu_umass_edu/zonghai/patientedu_image/xiong_2/UMASS-Advanced-NLP

# 创建logs目录
mkdir -p logs

# 打印环境信息
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPU: $SLURM_GPUS"
echo "Start Time: $(date)"
echo "=========================================="

# 检查数据文件是否存在
echo "Checking data files..."
files_needed=(
    "data/processed/biomed_mlm.jsonl"
    "data/processed/biomed_mlm_2.jsonl"
    "data/processed/finance_mlm1.jsonl"
    "data/processed/finance_mlm2.jsonl"
)

for file in "${files_needed[@]}"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: $file not found!"
        exit 1
    fi
done

echo "✅ All data files found. Starting MLM training with biomed+finance..."

# 运行训练（生物医学 + 金融混合）
echo "Running MLM training with biomedical + finance data..."
python src/models/pretraining_mlm.py \
    --config src/configs/mlm_bertgoemotions_biomed_finance.yaml

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "✅ Training completed successfully!"
    echo "End Time: $(date)"
    echo "Output directory: outputs/mlm_bert_goemotions_biomed_finance/"
    echo "=========================================="
    
    # 列出输出文件
    echo "Generated files:"
    ls -lh outputs/mlm_bert_goemotions_biomed_finance/
else
    echo "=========================================="
    echo "❌ Training failed!"
    echo "End Time: $(date)"
    echo "=========================================="
    exit 1
fi
