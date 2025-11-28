#!/bin/bash
# 监控所有MLM训练任务的状态

echo "🔄 正在检查MLM训练任务状态... ($(date '+%Y-%m-%d %H:%M:%S'))"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════════"

# 检查三个任务
JOBS=(49322623 49325785 49325786)
NAMES=("生物医学only" "生物+金融混合" "金融only")

for i in "${!JOBS[@]}"; do
    jobid=${JOBS[$i]}
    name=${NAMES[$i]}
    
    state=$(squeue -j $jobid --format=%T -h 2>/dev/null)
    reason=$(squeue -j $jobid --format=%r -h 2>/dev/null)
    time_left=$(squeue -j $jobid --format=%L -h 2>/dev/null)
    priority=$(squeue -j $jobid --format=%Y -h 2>/dev/null)
    
    if [ -z "$state" ]; then
        state="UNKNOWN"
    fi
    
    echo ""
    echo "📌 任务 $((i+1)): Job $jobid ($name)"
    echo "   状态: $state | 优先级: $priority | 剩余时间: $time_left"
    
    if [ "$state" == "RUNNING" ]; then
        echo "   ✅ 正在运行！"
        # 如果运行中，尝试显示日志
        logfile="logs/train_${jobid}.log"
        if [ -f "$logfile" ]; then
            lines=$(wc -l < "$logfile")
            echo "   📝 日志行数: $lines"
            tail_lines=$(tail -3 "$logfile" | tr '\n' ' ')
            echo "   📄 最近: $tail_lines"
        fi
    elif [ "$state" == "PENDING" ]; then
        echo "   ⏳ 排队中... (原因: $reason)"
    elif [ "$state" == "COMPLETED" ]; then
        echo "   ✅ 已完成！"
    elif [ "$state" == "FAILED" ]; then
        echo "   ❌ 已失败"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "⚡ GPU队列统计:"
running=$(squeue -p gpu-preempt --format=%T -h | grep -c "^R$")
pending=$(squeue -p gpu-preempt --format=%T -h | grep -c "^PD$")
echo "   运行中: $running | 排队中: $pending"
echo ""
echo "💡 查看详细日志: tail -f logs/train_49322623.log"
echo "📊 完整队列状态: squeue -u \$USER"
