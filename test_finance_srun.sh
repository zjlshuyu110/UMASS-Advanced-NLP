#!/bin/bash
# 测试脚本：使用srun交互式运行金融only训练
# 用于验证资源配置是否可行

echo "🚀 启动金融only训练测试..."
echo "使用srun方式，资源配置: 1x 1080ti GPU, 64GB内存, 3小时"
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
if [ ! -f "data/processed/finance_mlm1.jsonl" ] || [ ! -f "data/processed/finance_mlm2.jsonl" ]; then
    echo "❌ 错误: 数据文件不存在！"
    exit 1
fi
echo "✅ 数据文件已找到"
echo ""

# 运行训练
echo "开始训练 (金融only)..."
python src/models/pretraining_mlm.py \
    --config src/configs/mlm_bertgoemotions_finance_only.yaml

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 训练成功完成！"
    echo "完成时间: $(date)"
    echo "=========================================="
    echo ""
    echo "✅ 模型已保存到: outputs/mlm_bert_goemotions_finance/"
    ls -lh outputs/mlm_bert_goemotions_finance/
else
    echo ""
    echo "=========================================="
    echo "❌ 训练失败"
    echo "失败时间: $(date)"
    echo "=========================================="
    exit 1
fi
