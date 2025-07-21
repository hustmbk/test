#!/usr/bin/env python3
"""
DeepSeek-V3 与 RetroInfer 集成测试脚本

此脚本用于测试 DeepSeek-V3 模型使用 RetroInfer 优化的效果，
包括内存使用、推理速度和生成质量的对比。
"""

import os
import sys
import json
import time
import torch
import argparse
import psutil
import nvidia_ml_py as nvml
from termcolor import colored
from transformers import AutoTokenizer

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from model_hub import DeepSeekModel


def get_gpu_memory_usage():
    """获取GPU内存使用情况"""
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    info = nvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "total": info.total / 1024**3,  # GB
        "used": info.used / 1024**3,    # GB
        "free": info.free / 1024**3     # GB
    }


def get_cpu_memory_usage():
    """获取CPU内存使用情况"""
    memory = psutil.virtual_memory()
    return {
        "total": memory.total / 1024**3,     # GB
        "used": memory.used / 1024**3,       # GB
        "available": memory.available / 1024**3  # GB
    }


def create_long_context_input(length=120000):
    """创建长上下文测试输入"""
    # 创建一个包含多个段落的长文本
    base_text = """
在深度学习的快速发展中，大语言模型（LLM）已经成为人工智能领域的核心技术之一。
这些模型通过处理海量文本数据，学习语言的复杂模式和结构，从而能够理解和生成人类语言。

然而，随着模型规模的不断增长和应用场景的复杂化，如何高效地进行推理成为了一个关键挑战。
特别是在处理长上下文时，传统的注意力机制会导致内存使用呈二次方增长，严重限制了模型的实用性。

RetroInfer 通过将 KV 缓存视为向量存储，并引入了创新的波索引（Wave Index）机制，
实现了 GPU 和 CPU 的协同计算，大幅提升了长上下文推理的效率。

DeepSeek-V3 模型进一步创新，引入了 MLA（Multi-head Latent Attention）机制，
通过低秩压缩将 KV 缓存压缩了 32 倍，同时保持了模型性能。

让我们深入探讨这些技术是如何工作的...
"""
    
    # 重复文本以达到目标长度
    # 注意：实际token数会因tokenizer而异，这里只是粗略估计
    estimated_tokens_per_char = 0.25  # 中文大约4个字符一个token
    target_chars = int(length / estimated_tokens_per_char)
    
    repeated_text = base_text
    while len(repeated_text) < target_chars:
        repeated_text += base_text
    
    # 添加一个具体的问题
    question = "\n\n基于以上内容，请详细解释 RetroInfer 和 DeepSeek-V3 的 MLA 机制是如何协同工作的？"
    
    return repeated_text[:target_chars] + question


def test_standard_attention(model_name, input_text, max_new_tokens=100):
    """测试标准Flash Attention"""
    print(colored("\n=== 测试标准 Flash Attention ===", "cyan", attrs=["bold"]))
    
    # 记录初始内存
    gpu_mem_start = get_gpu_memory_usage()
    cpu_mem_start = get_cpu_memory_usage()
    
    # 初始化模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DeepSeekModel(
        model_name=model_name,
        max_length=150000,  # 支持更长的序列
        dtype=torch.float16,
        device_map="auto"
    )
    
    # Tokenize输入
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=120000)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print(f"输入长度: {input_ids.shape[1]} tokens")
    
    # 生成配置
    config = {
        "Full_Flash_Attn": {}
    }
    
    # 开始推理
    start_time = time.time()
    
    outputs = model.generate(
        attention_type="Full_Flash_Attn",
        inputs_ids=input_ids.to(model.layers[0].device),
        attention_masks=attention_mask.to(model.layers[0].device),
        max_new_length=max_new_tokens,
        attn_config=config
    )
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    # 记录结束内存
    gpu_mem_end = get_gpu_memory_usage()
    cpu_mem_end = get_cpu_memory_usage()
    
    # 解码输出
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    results = {
        "method": "Full_Flash_Attn",
        "input_tokens": input_ids.shape[1],
        "output_tokens": max_new_tokens,
        "inference_time": inference_time,
        "tokens_per_second": max_new_tokens / inference_time,
        "gpu_memory_used": gpu_mem_end["used"] - gpu_mem_start["used"],
        "cpu_memory_used": cpu_mem_end["used"] - cpu_mem_start["used"],
        "generated_text": generated_text[-500:]  # 只保留最后500个字符
    }
    
    return results


def test_retroinfer_mla(model_name, input_text, max_new_tokens=100):
    """测试RetroInfer + MLA"""
    print(colored("\n=== 测试 RetroInfer + MLA ===", "cyan", attrs=["bold"]))
    
    # 记录初始内存
    gpu_mem_start = get_gpu_memory_usage()
    cpu_mem_start = get_cpu_memory_usage()
    
    # 初始化模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DeepSeekModel(
        model_name=model_name,
        max_length=150000,
        dtype=torch.float16,
        device_map="auto"
    )
    
    # Tokenize输入
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=120000)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print(f"输入长度: {input_ids.shape[1]} tokens")
    
    # 加载RetroInfer配置
    config_path = os.path.join(PROJECT_ROOT, "config/DeepSeek-V3.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # 动态调整RetroInfer参数
    context_len = input_ids.shape[1]
    n_clusters = max(int(context_len/16), 1)
    n_segments = max(int(context_len/8192), 1)
    
    # 调整为最近的(n_segments*32)的倍数
    lower = (n_clusters // (n_segments*32)) * (n_segments*32)
    upper = lower + (n_segments*32)
    n_clusters = lower if abs(n_clusters - lower) <= abs(n_clusters - upper) else upper
    nprobe = max(int(n_clusters*0.018), 1)
    
    # 更新配置
    retroinfer_config = config["DeepSeek-V3"]["RetroInfer"].copy()
    retroinfer_config.update({
        "n_centroids": n_clusters,
        "n_segment": n_segments,
        "nprobe": nprobe,
        "cache_cluster_num": nprobe * 3,
        "max_compute_cluster_num": max(int(n_clusters/4), nprobe)
    })
    
    attn_config = {
        "RetroInfer": retroinfer_config,
        "MLA": config["DeepSeek-V3"]["MLA"]
    }
    
    print(f"RetroInfer 配置:")
    print(f"  - n_centroids: {n_clusters}")
    print(f"  - n_segments: {n_segments}")
    print(f"  - nprobe: {nprobe}")
    
    # 开始推理
    start_time = time.time()
    
    outputs = model.generate(
        attention_type="RetroInfer",
        inputs_ids=input_ids.to(model.layers[0].device),
        attention_masks=attention_mask.to(model.layers[0].device),
        max_new_length=max_new_tokens,
        attn_config=attn_config
    )
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    # 记录结束内存
    gpu_mem_end = get_gpu_memory_usage()
    cpu_mem_end = get_cpu_memory_usage()
    
    # 解码输出
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    results = {
        "method": "RetroInfer + MLA",
        "input_tokens": input_ids.shape[1],
        "output_tokens": max_new_tokens,
        "inference_time": inference_time,
        "tokens_per_second": max_new_tokens / inference_time,
        "gpu_memory_used": gpu_mem_end["used"] - gpu_mem_start["used"],
        "cpu_memory_used": cpu_mem_end["used"] - cpu_mem_start["used"],
        "n_clusters": n_clusters,
        "nprobe": nprobe,
        "generated_text": generated_text[-500:]
    }
    
    return results


def compare_results(results_standard, results_retroinfer):
    """比较两种方法的结果"""
    print(colored("\n=== 性能对比 ===", "green", attrs=["bold"]))
    print("=" * 80)
    
    # 格式化输出对比结果
    metrics = [
        ("方法", "method"),
        ("输入tokens", "input_tokens"),
        ("输出tokens", "output_tokens"),
        ("推理时间(秒)", "inference_time"),
        ("生成速度(tokens/秒)", "tokens_per_second"),
        ("GPU内存增量(GB)", "gpu_memory_used"),
        ("CPU内存增量(GB)", "cpu_memory_used")
    ]
    
    for metric_name, metric_key in metrics:
        std_value = results_standard.get(metric_key, "N/A")
        retro_value = results_retroinfer.get(metric_key, "N/A")
        
        if isinstance(std_value, float):
            std_str = f"{std_value:.2f}"
            retro_str = f"{retro_value:.2f}"
            
            # 计算改进百分比
            if metric_key in ["inference_time", "gpu_memory_used", "cpu_memory_used"]:
                # 越小越好的指标
                improvement = (std_value - retro_value) / std_value * 100
                improvement_str = f"↓ {improvement:.1f}%" if improvement > 0 else f"↑ {-improvement:.1f}%"
            else:
                # 越大越好的指标
                improvement = (retro_value - std_value) / std_value * 100
                improvement_str = f"↑ {improvement:.1f}%" if improvement > 0 else f"↓ {-improvement:.1f}%"
        else:
            std_str = str(std_value)
            retro_str = str(retro_value)
            improvement_str = ""
        
        print(f"{metric_name:<20} | 标准: {std_str:<15} | RetroInfer: {retro_str:<15} | {improvement_str}")
    
    # 特殊参数（仅RetroInfer有）
    if "n_clusters" in results_retroinfer:
        print(f"\nRetroInfer 特殊参数:")
        print(f"  - 聚类数量: {results_retroinfer['n_clusters']}")
        print(f"  - 探测数量: {results_retroinfer['nprobe']}")
    
    print("\n" + "=" * 80)
    
    # 总结
    print(colored("\n总结:", "yellow", attrs=["bold"]))
    
    speed_improvement = (results_retroinfer["tokens_per_second"] - results_standard["tokens_per_second"]) / results_standard["tokens_per_second"] * 100
    memory_saving = (results_standard["gpu_memory_used"] - results_retroinfer["gpu_memory_used"]) / results_standard["gpu_memory_used"] * 100
    
    print(f"• 推理速度提升: {speed_improvement:.1f}%")
    print(f"• GPU内存节省: {memory_saving:.1f}%")
    print(f"• MLA压缩比: 32x (理论值)")
    print(f"• 激活参数比例: 5.5% (仅激活37B/671B参数)")


def main():
    parser = argparse.ArgumentParser(description="测试DeepSeek-V3与RetroInfer的集成效果")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-V3",
                        help="模型名称或路径")
    parser.add_argument("--context-length", type=int, default=100000,
                        help="输入上下文长度（tokens）")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="生成的新token数量")
    parser.add_argument("--compare", action="store_true",
                        help="是否进行对比测试")
    
    args = parser.parse_args()
    
    print(colored("DeepSeek-V3 + RetroInfer 集成测试", "magenta", attrs=["bold"]))
    print("=" * 80)
    print(f"模型: {args.model}")
    print(f"目标上下文长度: {args.context_length} tokens")
    print(f"生成长度: {args.max_new_tokens} tokens")
    print("=" * 80)
    
    # 创建测试输入
    print(colored("\n准备测试数据...", "yellow"))
    test_input = create_long_context_input(args.context_length)
    print(f"创建了约 {len(test_input)} 字符的输入文本")
    
    if args.compare:
        # 对比测试
        try:
            # 测试标准方法
            results_standard = test_standard_attention(
                args.model, test_input, args.max_new_tokens
            )
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            time.sleep(5)  # 等待内存释放
            
            # 测试RetroInfer方法
            results_retroinfer = test_retroinfer_mla(
                args.model, test_input, args.max_new_tokens
            )
            
            # 比较结果
            compare_results(results_standard, results_retroinfer)
            
        except Exception as e:
            print(colored(f"\n错误: {str(e)}", "red"))
            print("提示: 如果遇到内存不足，请尝试减小 --context-length 参数")
    
    else:
        # 仅测试RetroInfer
        try:
            results = test_retroinfer_mla(
                args.model, test_input, args.max_new_tokens
            )
            
            print(colored("\n测试结果:", "green", attrs=["bold"]))
            print(f"推理时间: {results['inference_time']:.2f} 秒")
            print(f"生成速度: {results['tokens_per_second']:.2f} tokens/秒")
            print(f"GPU内存使用: {results['gpu_memory_used']:.2f} GB")
            print(f"CPU内存使用: {results['cpu_memory_used']:.2f} GB")
            
        except Exception as e:
            print(colored(f"\n错误: {str(e)}", "red"))
            import traceback
            traceback.print_exc()
    
    print(colored("\n测试完成！", "green", attrs=["bold"]))


if __name__ == "__main__":
    main()