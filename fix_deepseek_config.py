#!/usr/bin/env python3
"""
DeepSeek 配置兼容性处理工具

解决不同版本的 DeepSeek 模型配置差异问题
"""

import json
import os
from typing import Dict, Any

def get_deepseek_config_defaults(model_version: str = "v2") -> Dict[str, Any]:
    """
    获取 DeepSeek 模型的默认配置值
    
    Args:
        model_version: 模型版本 (v2, v2-lite, v3)
    
    Returns:
        默认配置字典
    """
    # 基础默认值
    defaults = {
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "use_cache": True,
    }
    
    # 版本特定的默认值
    if model_version == "v3":
        defaults.update({
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 1408,  # 根据论文
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
        })
    elif model_version == "v2":
        defaults.update({
            "num_experts": 160,
            "num_experts_per_tok": 6,
            "moe_intermediate_size": 1408,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
        })
    elif model_version == "v2-lite":
        defaults.update({
            "num_experts": 64,
            "num_experts_per_tok": 6,
            "moe_intermediate_size": 1408,
            "q_lora_rank": 512,
            "kv_lora_rank": 256,
            "qk_rope_head_dim": 64,
            "v_head_dim": 64,
        })
    
    return defaults


def merge_config_with_defaults(config: Dict[str, Any], model_version: str) -> Dict[str, Any]:
    """
    合并配置与默认值
    
    Args:
        config: 原始配置
        model_version: 模型版本
    
    Returns:
        合并后的配置
    """
    defaults = get_deepseek_config_defaults(model_version)
    
    # 创建新配置，保留原始值，补充缺失的默认值
    merged_config = config.copy()
    for key, value in defaults.items():
        if key not in merged_config:
            merged_config[key] = value
    
    return merged_config


def check_config_compatibility(config_path: str) -> bool:
    """
    检查配置文件兼容性
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        是否兼容
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        required_fields = [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "vocab_size",
        ]
        
        for field in required_fields:
            if field not in config:
                print(f"❌ Missing required field: {field}")
                return False
        
        print("✅ Config file is compatible")
        return True
        
    except Exception as e:
        print(f"❌ Error checking config: {e}")
        return False


def fix_config_file(config_path: str, model_version: str, output_path: str = None):
    """
    修复配置文件，添加缺失的字段
    
    Args:
        config_path: 原始配置文件路径
        model_version: 模型版本
        output_path: 输出路径（如果为 None，则覆盖原文件）
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 合并默认值
        fixed_config = merge_config_with_defaults(config, model_version)
        
        # 保存修复后的配置
        if output_path is None:
            output_path = config_path
        
        with open(output_path, 'w') as f:
            json.dump(fixed_config, f, indent=2)
        
        print(f"✅ Config file fixed and saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error fixing config: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepSeek 配置兼容性工具")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--version", type=str, default="v2", 
                       choices=["v2", "v2-lite", "v3"], help="模型版本")
    parser.add_argument("--fix", action="store_true", help="修复配置文件")
    parser.add_argument("--output", type=str, help="输出路径（仅在修复时使用）")
    
    args = parser.parse_args()
    
    if args.fix:
        fix_config_file(args.config, args.version, args.output)
    else:
        check_config_compatibility(args.config)