#!/bin/bash
# 运行DeepSeek-V3与RetroInfer测试的脚本

echo "=========================================="
echo "DeepSeek-V3 + RetroInfer 测试脚本"
echo "=========================================="

# 检查CUDA是否可用
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA设备数: {torch.cuda.device_count()}')"

echo ""
echo "请选择测试类型:"
echo "1. 快速测试 (10K tokens)"
echo "2. 标准测试 (50K tokens)"
echo "3. 长序列测试 (100K tokens)"
echo "4. 极限测试 (120K tokens)"
echo "5. 对比测试 (50K tokens, 标准 vs RetroInfer)"

read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        echo "运行快速测试..."
        python test_deepseek_retroinfer.py \
            --context-length 10000 \
            --max-new-tokens 50
        ;;
    2)
        echo "运行标准测试..."
        python test_deepseek_retroinfer.py \
            --context-length 50000 \
            --max-new-tokens 100
        ;;
    3)
        echo "运行长序列测试..."
        python test_deepseek_retroinfer.py \
            --context-length 100000 \
            --max-new-tokens 100
        ;;
    4)
        echo "运行极限测试..."
        echo "警告: 此测试需要大量内存!"
        python test_deepseek_retroinfer.py \
            --context-length 120000 \
            --max-new-tokens 100
        ;;
    5)
        echo "运行对比测试..."
        echo "警告: 此测试会运行两次，需要更多时间和内存!"
        python test_deepseek_retroinfer.py \
            --context-length 50000 \
            --max-new-tokens 100 \
            --compare
        ;;
    *)
        echo "无效选项!"
        exit 1
        ;;
esac

echo ""
echo "测试完成!"