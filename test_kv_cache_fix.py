#!/usr/bin/env python3
"""
测试KV缓存修复的简单脚本
"""

import os
import sys
import torch

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

def test_flash_attn_cache():
    """测试flash_attn_cache是否能正确初始化"""
    print("测试 flash_attn_cache 初始化...")
    
    try:
        from cache_hub import flash_attn_cache
        
        # 创建测试参数
        valid_start = [0, 0]  # 假设2个批次
        layer_num = 4
        batch_size = 2
        max_length = 100
        num_key_value_heads = 8
        num_heads = 16
        head_dim = 64
        dtype = torch.float16
        layer_mapping = {str(i): 'cuda:0' for i in range(layer_num)}
        num_gpus = 1
        model_size = 8  # 8GB模型大小
        
        # 尝试初始化缓存
        cache = flash_attn_cache(
            valid_start=valid_start,
            layer_num=layer_num,
            batch_size=batch_size,
            max_length=max_length,
            num_key_value_heads=num_key_value_heads,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            layer_mapping=layer_mapping,
            num_gpus=num_gpus,
            model_size=model_size
        )
        
        print("✓ flash_attn_cache 初始化成功!")
        print(f"  - 层数: {cache.layer_num}")
        print(f"  - 批次大小: {cache.batch_size}")
        print(f"  - 最大长度: {cache.max_length}")
        print(f"  - 模型大小: {cache.model_size}GB")
        print(f"  - 空闲内存: {cache.free_memory:.2f}GB")
        
        return True
        
    except Exception as e:
        print(f"✗ flash_attn_cache 初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_deepseek_init():
    """测试DeepSeek模型初始化（不加载实际权重）"""
    print("\n测试 DeepSeek 模型初始化...")
    
    try:
        # 先创建一个模拟的配置类
        class MockConfig:
            def __init__(self):
                self.num_hidden_layers = 4
                self.hidden_size = 512
                self.vocab_size = 32000
                self.num_attention_heads = 16
                self.num_key_value_heads = 8
                self.kv_lora_rank = 128
                self.q_lora_rank = 256
                self.qk_rope_head_dim = 32
                self.v_head_dim = 32
                self.rms_norm_eps = 1e-6
                self.num_experts = 8
                self.num_experts_per_tok = 2
                self.moe_intermediate_size = 1024
                self.intermediate_size = 2048
                self.rope_theta = 10000.0
        
        # 模拟tokenizer
        class MockTokenizer:
            def __init__(self):
                self.eos_token = "</s>"
                self.pad_token = "</s>"
        
        # 测试DeepSeek初始化（只测试关键参数）
        from model_hub.deepseek import DeepSeekModel
        
        # 创建一个最小的测试实例
        model = object.__new__(DeepSeekModel)  # 不调用__init__
        model.model_name = "test"
        model.max_length = 1000
        model.dtype = torch.float16
        model.device_map = "cuda:0"
        model.model_version = "v2-lite"
        
        # 设置配置
        model.config = MockConfig()
        model.tokenizer = MockTokenizer()
        
        # 测试关键参数设置
        model.num_layers = 4
        model.hidden_size = 512
        model.vocab_size = 32000
        model.kv_lora_rank = 128
        model.q_lora_rank = 256
        model.num_experts = 8
        model.num_experts_per_tok = 2
        model.model_size_gb = 8  # 这是我们修复的关键参数
        
        # 设置模型参数（用于缓存初始化）
        model.batch_size = 1
        model.max_new_length = 50
        model.num_heads = 16
        model.num_key_value_heads = 8
        model.head_dim = 32
        model.num_gpus = 1
        model.layer_mapping = {str(i): 'cuda:0' for i in range(4)}
        
        print("✓ DeepSeek 模型参数设置成功!")
        print(f"  - 模型版本: {model.model_version}")
        print(f"  - 层数: {model.num_layers}")
        print(f"  - 隐藏维度: {model.hidden_size}")
        print(f"  - KV压缩维度: {model.kv_lora_rank}")
        print(f"  - 模型大小: {model.model_size_gb}GB")
        
        # 测试init_kv_cache方法
        print("\n测试 init_kv_cache 方法...")
        
        real_input_length = 50
        valid_start = [0]  # 1个批次
        
        # 调用init_kv_cache方法
        model.init_kv_cache(real_input_length, valid_start)
        
        print("✓ KV缓存初始化成功!")
        print(f"  - 缓存类型: {type(model.kv_cache).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ DeepSeek 初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== KV缓存修复测试 ===\n")
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB\n")
    else:
        print("⚠ CUDA不可用，使用CPU模式\n")
    
    success_count = 0
    total_tests = 2
    
    # 测试1: flash_attn_cache初始化
    if test_flash_attn_cache():
        success_count += 1
    
    # 测试2: DeepSeek模型初始化
    if test_deepseek_init():
        success_count += 1
    
    # 总结
    print(f"\n=== 测试结果 ===")
    print(f"通过: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 所有测试通过! model_size参数修复成功!")
    else:
        print("❌ 部分测试失败，需要进一步调试")

if __name__ == "__main__":
    main()