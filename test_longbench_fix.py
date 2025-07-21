#!/usr/bin/env python3
"""
测试 LongBench 数据集加载是否正常工作
"""

import sys
from datasets import load_dataset

def test_longbench_loading():
    """测试加载 LongBench 数据集"""
    try:
        print("正在测试加载 LongBench 数据集...")
        
        # 测试加载一个小的数据集
        dataset_name = "qasper"
        print(f"尝试加载数据集: THUDM/LongBench/{dataset_name}")
        
        # 使用 trust_remote_code=True
        data = load_dataset('THUDM/LongBench', dataset_name, split='test', trust_remote_code=True)
        
        print(f"✓ 成功加载数据集！")
        print(f"  数据集大小: {len(data)} 条记录")
        
        # 显示第一条数据的键
        if len(data) > 0:
            print(f"  数据字段: {list(data[0].keys())}")
            
        return True
        
    except Exception as e:
        print(f"✗ 加载失败: {str(e)}")
        return False

def test_without_trust_remote_code():
    """测试不使用 trust_remote_code 的情况"""
    try:
        print("\n测试不使用 trust_remote_code 参数...")
        data = load_dataset('THUDM/LongBench', 'qasper', split='test')
        print("✗ 意外成功 - 应该失败!")
        return False
    except Exception as e:
        print(f"✓ 预期的失败: {str(e)}")
        return True

if __name__ == "__main__":
    print("=== LongBench 数据集加载测试 ===\n")
    
    # 测试修复后的版本
    success1 = test_longbench_loading()
    
    # 测试未修复的版本（应该失败）
    success2 = test_without_trust_remote_code()
    
    print("\n=== 测试总结 ===")
    if success1:
        print("✓ 修复有效：使用 trust_remote_code=True 可以正常加载数据集")
    else:
        print("✗ 修复无效：即使使用 trust_remote_code=True 仍然无法加载")
    
    sys.exit(0 if success1 else 1)