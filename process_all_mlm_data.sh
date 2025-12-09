#!/bin/bash
#
# 重新处理所有 Stage 1 (MLM) 数据
# 运行方式: srun --partition=gpu-preempt --gres=gpu:2080ti:1 --mem=64GB --time=4:00:00 bash process_all_mlm_data.sh
#

set -e  # 遇到错误立即退出

# 激活 conda 环境
source /work/pi_hongyu_umass_edu/xiongluo_umass_edu/miniconda3/etc/profile.d/conda.sh
conda activate mlm_training

cd /project/pi_hongyu_umass_edu/zonghai/patientedu_image/xiong_2/UMASS-Advanced-NLP

echo "=========================================="
echo "清理旧数据..."
echo "=========================================="
rm -f data/processed/unlabel_*.jsonl

echo ""
echo "=========================================="
echo "1/5: 处理 BioASQ 数据 (unlabel_bio_1)"
echo "=========================================="
python src/data/unlabel_bio_1.py

echo ""
echo "=========================================="
echo "2/5: 处理 biomedical_en-de 数据 (unlabel_bio_2)"
echo "=========================================="
python src/data/unlabel_bio_2.py

echo ""
echo "=========================================="
echo "3/5: 处理 medical_text 数据 (unlabel_bio_3)"
echo "=========================================="
python src/data/unlabel_bio_3.py

echo ""
echo "=========================================="
echo "4/5: 处理 finance_news 数据 (unlabel_financial_1)"
echo "=========================================="
python src/data/unlabel_financial_1.py

echo ""
echo "=========================================="
echo "5/5: 处理 finance_news + phrasebank 数据 (unlabel_financial_2)"
echo "=========================================="
python src/data/unlabel_financial_2.py

echo ""
echo "=========================================="
echo "数据处理完成！统计信息："
echo "=========================================="
for file in data/processed/unlabel_*.jsonl; do
    if [ -f "$file" ]; then
        count=$(wc -l < "$file")
        size=$(du -h "$file" | cut -f1)
        echo "  $(basename $file): $count 条数据 ($size)"
    fi
done

echo ""
echo "=========================================="
echo "生物医学总计："
wc -l data/processed/unlabel_bio_*.jsonl | tail -1

echo ""
echo "金融总计："
wc -l data/processed/unlabel_financial_*.jsonl | tail -1

echo ""
echo "✅ 所有数据处理完成！"
