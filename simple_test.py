import os
import sys
import json
import math
import torch
import argparse
import random
import numpy as np
from termcolor import colored
from transformers import AutoTokenizer

# [修正] 确保项目根目录被正确添加到路径中
# 这样可以找到 model_hub 目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# [官方做法] 从 model_hub 导入重构后的模型类
# 我们假设 deepseek_v2_model.py 和其他模型文件都在 model_hub 目录下
try:
    from model_hub import DeepSeekV2Model
    from model_hub import LlamaModel, QwenModel # 如果需要，取消注释
except ImportError:
    print(colored("错误: 无法从 'model_hub' 导入模型。请确保您的文件结构如下:", 'red'))
    print("your_project/")
    print("├── test.py (此文件)")
    print("└── model_hub/")
    print("    ├── __init__.py")
    print("    └── deepseek_v2_model.py")
    sys.exit(1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Test example with official DeepSeek-V2 MLA support")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gen_len", type=int, default=100, help="Generation length")
    parser.add_argument("--device", type=str, default="auto", help="Device map ('auto', 'cuda:0', etc.)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Dtype")
    
    # [修正] 默认使用 Full_Flash_Attn，因为这是针对DeepSeek-V2的官方推荐做法
    parser.add_argument("--attn_type", type=str, default="Full_Flash_Attn",
                        choices=["Full_Flash_Attn", "RetroInfer"], help="Attention method")
    
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-V2-Lite-Chat",
                        help="HuggingFace model name or local path")
    parser.add_argument("--model_type", type=str, default="auto",
                        choices=["auto", "llama", "qwen", "deepseek"],
                        help="Force model type (auto-detect if 'auto')")
    
    parser.add_argument("--data_path", type=str, default="", help="Input json file path")
    parser.add_argument("--test_mla_fix", action="store_true", help="Run diagnostic test for MLA architecture.")
    
    args = parser.parse_args()
    return args


def detect_model_type(model_name):
    """Auto-detect model type from model name"""
    model_name_lower = model_name.lower()
    if any(keyword in model_name_lower for keyword in ['deepseek', 'deepseek-v2', 'deepseek-coder']):
        return 'deepseek'
    elif any(keyword in model_name_lower for keyword in ['llama', 'llama-3', 'llama-2']):
        return 'llama'
    elif any(keyword in model_name_lower for keyword in ['qwen', 'qwen2']):
        return 'qwen'
    else:
        print(colored(f"⚠️  Cannot auto-detect model type for {model_name}, assuming Llama.", 'yellow'))
        return 'llama'

def generate_config(attn_type, context_len):
    """
    [修正] 简化配置生成。
    对于 Full_Flash_Attn，不需要任何特殊配置。
    对于 RetroInfer，保留其计算逻辑。
    """
    if attn_type == "Full_Flash_Attn":
        print(colored("Using Full_Flash_Attn: No special config needed.", 'cyan'))
        return None

    # RetroInfer的配置逻辑保持不变
    if attn_type == 'RetroInfer':
        print(colored(f"Generating config for RetroInfer with context length {context_len}...", 'cyan'))
        n_clusters = max(int(context_len / 16), 1)
        n_segments = max(int(context_len / 8192), 1)
        segment_multiple = n_segments * 32
        if segment_multiple > 0:
            lower = (n_clusters // segment_multiple) * segment_multiple
            upper = lower + segment_multiple
            n_clusters = lower if abs(n_clusters - lower) <= abs(n_clusters - upper) else upper
        nprobe = max(int(n_clusters * 0.018), 1)
        
        config = {
            "RetroInfer": {
                "n_centroids": n_clusters,
                "n_segment": n_segments,
                "nprobe": nprobe,
                "cache_cluster_num": nprobe * 3,
                "max_compute_cluster_num": max(int(n_clusters / 4), nprobe)
            }
        }
        print(json.dumps(config[attn_type], indent=2))
        return config
    
    return None


def test_mla_architecture(llm):
    """[修正] 基于模型加载后的config对象来诊断MLA架构，更稳定可靠。"""
    if not isinstance(llm, DeepSeekV2Model):
        print(colored("⚠️  Not a DeepSeekV2Model instance, skipping MLA test.", 'yellow'))
        return

    print(colored("\n🧪 Running MLA Architecture Sanity Check:", 'cyan'))
    config = llm.config

    try:
        # 测试1：验证KV头数（MLA中应等于Q头数）
        if config.num_key_value_heads == config.num_attention_heads:
            print(colored(f"✅ Test 1 Passed: num_key_value_heads ({config.num_key_value_heads}) == num_attention_heads ({config.num_attention_heads})", 'green'))
        else:
            print(colored(f"❌ Test 1 Failed: KV heads ({config.num_key_value_heads}) != Q heads ({config.num_attention_heads})", 'red'))

        # 测试2：验证维度（不同的Q/K和V维度是MLA的关键特征）
        q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        v_head_dim = config.v_head_dim
        if q_head_dim == 192 and v_head_dim == 128:
            print(colored(f"✅ Test 2 Passed: Q/K head dim is {q_head_dim}D, V head dim is {v_head_dim}D.", 'green'))
        else:
            print(colored(f"❌ Test 2 Failed: Unexpected dimensions Q/K={q_head_dim}D, V={v_head_dim}D.", 'red'))

        # 测试3：验证KV LoRA Rank存在
        if hasattr(config, 'kv_lora_rank') and config.kv_lora_rank > 0:
            print(colored(f"✅ Test 3 Passed: kv_lora_rank is present ({config.kv_lora_rank}).", 'green'))
        else:
            print(colored("❌ Test 3 Failed: kv_lora_rank not found or is zero.", 'red'))

        print(colored("\n📋 MLA Architecture Summary:", 'white'))
        print("  - MLA (Multi-head Latent Attention) uses low-rank projection to compress the KV cache.")
        print("  - It does NOT reduce the number of heads; it reduces the dimension of what's stored.")
        print("  - The result is a >90% reduction in KV cache memory with comparable performance.")
        
    except AttributeError as e:
        print(colored(f"❌ Diagnostic test failed: A required config attribute is missing: {e}", 'red'))
    print()


def load_model(model_name, model_type, max_len, dtype, device):
    """加载模型，根据类型选择正确的类。"""
    if model_type == "auto":
        model_type = detect_model_type(model_name)
    
    print(colored(f"Loading model of type '{model_type}': {model_name}", 'cyan'))
    
    # [官方做法] 当模型是deepseek时，实例化我们重构后的DeepSeekV2Model
    if model_type == 'deepseek':
        return DeepSeekV2Model(
            model_name,
            max_length=max_len,
            dtype=dtype,
            device_map=device
        )
    # elif model_type == 'llama':
    #     return LlamaModel(...)
    # elif model_type == 'qwen':
    #     return QwenModel(...)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Please implement or choose another model.")


def main():
    args = parse_args()
    set_seed(2025)

    dtype = torch.float16 if args.dtype == 'fp16' else torch.bfloat16

    print(colored("="*50, 'cyan'))
    print(colored("🚀 Starting DeepSeek-V2 Test (Official Implementation)", 'cyan'))
    print(f"   Model: {args.model_name}")
    print(f"   Attention: {args.attn_type}")
    print(f"   Batch Size: {args.batch_size}")
    print(colored("="*50, 'cyan'))

    # 加载模型
    max_len = 2048 # 设定一个合理的默认最大长度
    llm = load_model(args.model_name, args.model_type, max_len, dtype, args.device)
    
    # 如果需要，运行架构诊断
    if args.test_mla_fix and args.model_type in ['deepseek', 'auto']:
        test_mla_architecture(llm)

    # 加载并准备数据
    if not args.data_path:
        print(colored("No data path provided, using default prompt.", 'yellow'))
        prompts = ["Write a short story about a robot who discovers music."] * args.batch_size
    else:
        try:
            with open(args.data_path, 'r') as f:
                data = json.load(f)
            prompts = [item['input'] for item in data] * math.ceil(args.batch_size / len(data))
            prompts = prompts[:args.batch_size]
        except Exception as e:
            print(colored(f"Error loading data file: {e}. Using default prompt.", 'red'))
            prompts = ["Hello, how are you?"] * args.batch_size

    # [修正] 对于Full_Flash_Attn模式，不再需要复杂的attn_config
    # generate_config函数现在只为RetroInfer返回有效配置
    attn_config = generate_config(args.attn_type, 1024) # 传入一个示例长度

    # --- 执行推理 ---
    # [官方做法] 调用模型上更简洁、面向用户的generate方法。
    # 模型内部会处理分词、缓存设置和完整的prefill/decode循环。
    for i, prompt in enumerate(prompts):
        print(colored(f"\n--- Generating for Sample {i+1}/{args.batch_size} ---", 'yellow'))
        print(f"Prompt: {prompt}")
        
        try:
            # llm.generate现在是主要的推理入口
            response = llm.generate(
                inputs=prompt,
                max_new_tokens=args.gen_len
            )
            print(colored("Generated Response:", 'green'))
            print(response)
        except Exception as e:
            print(colored(f"❌ Generation failed for sample {i+1}: {e}", 'red'))
            import traceback
            traceback.print_exc()

    print(colored("\n🎉 Test completed!", 'cyan'))


if __name__ == "__main__":
    main()
