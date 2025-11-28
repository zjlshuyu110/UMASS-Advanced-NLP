#!/bin/bash
# srun 脚本：生物医学 only 训练
# 用法: srun -p gpu -G 1 --constraint=2080ti --mem=64G -t 3:00:00 --pty bash srun_train_biomed_only.sh

echo "🚀 启动生物医学 only 训练..."
echo "资源配置: 1x 2080ti GPU, 64GB内存, 3小时"
echo ""

# 激活环境
module load conda/latest
conda activate mlm_training

cd /project/pi_hongyu_umass_edu/zonghai/patientedu_image/xiong_2/UMASS-Advanced-NLP

# 创建logs目录
mkdir -p logs

echo "=========================================="
echo "开始时间: $(date)"
echo "=========================================="
echo ""

# 检查数据文件
echo "检查数据文件..."
if [ ! -f "data/processed/biomed_mlm.jsonl" ] || [ ! -f "data/processed/biomed_mlm_2.jsonl" ]; then
    echo "❌ 错误: 数据文件不存在！"
    exit 1
fi
echo "✅ 数据文件已找到"
echo ""

# 运行训练
echo "开始训练 (生物医学 only)..."
python src/models/pretraining_mlm.py \
    --config src/configs/mlm_bertgoemotions_biomed_only.yaml

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 训练成功完成！"
    echo "完成时间: $(date)"
    echo "=========================================="
    echo ""
    echo "✅ 模型已保存到: outputs/mlm_bert_goemotions_biomed/"
    ls -lh outputs/mlm_bert_goemotions_biomed/
else
    echo ""
    echo "=========================================="
    echo "❌ 训练失败"
    echo "完成时间: $(date)"
    echo "=========================================="
    exit 1
fi
