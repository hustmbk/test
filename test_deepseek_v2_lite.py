#!/usr/bin/env python3
"""
DeepSeek-V2-Lite Specific Test Script

Optimized testing for the lightweight DeepSeek-V2-Lite model (15.7B total, 2.8B active)
This script is tailored for resource-constrained environments.
"""

import os
import sys
import json
import time
import torch
import argparse
import psutil
from termcolor import colored
from transformers import AutoTokenizer

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from model_hub import DeepSeekModel


def print_v2_lite_info():
    """Print DeepSeek-V2-Lite model information"""
    print(colored("\n📊 DeepSeek-V2-Lite Model Specifications:", "cyan", attrs=["bold"]))
    print("=" * 60)
    print("• Total Parameters: 15.7B")
    print("• Active Parameters: 2.8B (17.8% activation ratio)")
    print("• Architecture: MoE with MLA")
    print("• Hidden Layers: 27")
    print("• Hidden Size: 2048")
    print("• MoE Experts: 64 total, 6 activated per token")
    print("• KV Compression: 16x (optimized for efficiency)")
    print("• Memory Usage: ~8GB GPU (FP16)")
    print("=" * 60)


def test_basic_generation():
    """Test basic text generation"""
    print(colored("\n🚀 Test 1: Basic Generation", "yellow", attrs=["bold"]))
    
    # Simple prompt
    prompt = "人工智能的未来发展趋势包括"
    
    # Initialize model
    model = DeepSeekModel(
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        max_length=32768,  # V2-Lite optimized for shorter contexts
        dtype=torch.float16,
        device_map="auto",
        model_version="v2-lite"
    )
    
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate
    start_time = time.time()
    outputs = model.generate(
        attention_type="RetroInfer",
        inputs_ids=inputs["input_ids"].to(model.layers[0].device),
        attention_masks=inputs["attention_mask"].to(model.layers[0].device),
        max_new_length=100,
        attn_config=get_v2_lite_config()
    )
    end_time = time.time()
    
    # Decode and display
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nInput: {prompt}")
    print(f"Output: {generated}")
    print(f"Generation time: {end_time - start_time:.2f}s")
    print(f"Speed: {100/(end_time - start_time):.2f} tokens/s")
    
    return True


def test_memory_efficiency():
    """Test memory efficiency with different context lengths"""
    print(colored("\n💾 Test 2: Memory Efficiency", "yellow", attrs=["bold"]))
    
    context_lengths = [1000, 5000, 10000, 20000]
    results = []
    
    for length in context_lengths:
        print(f"\nTesting {length} tokens...")
        
        # Create input
        text = "这是一个测试文本。" * (length // 10)
        
        # Get memory before
        mem_before = psutil.Process().memory_info().rss / 1024**3  # GB
        
        try:
            # Initialize model
            model = DeepSeekModel(
                model_name="deepseek-ai/DeepSeek-V2-Lite",
                max_length=length + 100,
                dtype=torch.float16,
                device_map="auto",
                model_version="v2-lite"
            )
            
            # Tokenize
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=length)
            
            # Generate short output
            start_time = time.time()
            outputs = model.generate(
                attention_type="RetroInfer",
                inputs_ids=inputs["input_ids"].to(model.layers[0].device),
                attention_masks=inputs["attention_mask"].to(model.layers[0].device),
                max_new_length=10,
                attn_config=get_v2_lite_config(length)
            )
            end_time = time.time()
            
            # Get memory after
            mem_after = psutil.Process().memory_info().rss / 1024**3
            
            results.append({
                "length": length,
                "memory_used": mem_after - mem_before,
                "time": end_time - start_time,
                "success": True
            })
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            results.append({
                "length": length,
                "memory_used": 0,
                "time": 0,
                "success": False,
                "error": str(e)
            })
    
    # Display results
    print("\n📊 Memory Efficiency Results:")
    print("-" * 60)
    print(f"{'Context Length':<15} {'Memory (GB)':<12} {'Time (s)':<10} {'Status':<10}")
    print("-" * 60)
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"{r['length']:<15} {r['memory_used']:<12.2f} {r['time']:<10.2f} {status:<10}")
    
    return results


def test_retroinfer_optimization():
    """Test RetroInfer optimization effectiveness"""
    print(colored("\n⚡ Test 3: RetroInfer Optimization", "yellow", attrs=["bold"]))
    
    # Medium-length context
    context = """
    DeepSeek-V2-Lite是DeepSeek系列中的轻量级版本，专为资源受限环境设计。
    该模型保持了MLA（Multi-head Latent Attention）和MoE（Mixture of Experts）
    的核心架构，但进行了针对性优化：
    
    1. 参数规模优化：总参数15.7B，激活参数仅2.8B
    2. 专家数量调整：64个专家，每次激活6个
    3. KV缓存压缩：16倍压缩率，平衡性能与内存
    4. 推理速度提升：相比完整版本，速度提升40%
    
    这些优化使得V2-Lite能够在消费级GPU上高效运行，
    同时保持优秀的生成质量。
    """ * 20  # Repeat to create longer context
    
    # Test with and without RetroInfer
    model_name = "deepseek-ai/DeepSeek-V2-Lite"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=10000)
    
    print(f"Context length: {inputs['input_ids'].shape[1]} tokens")
    
    # Test 1: With RetroInfer
    print("\n🔧 With RetroInfer:")
    model1 = DeepSeekModel(
        model_name=model_name,
        max_length=15000,
        dtype=torch.float16,
        device_map="auto",
        model_version="v2-lite"
    )
    
    start1 = time.time()
    out1 = model1.generate(
        attention_type="RetroInfer",
        inputs_ids=inputs["input_ids"].to(model1.layers[0].device),
        attention_masks=inputs["attention_mask"].to(model1.layers[0].device),
        max_new_length=50,
        attn_config=get_v2_lite_config(inputs['input_ids'].shape[1])
    )
    end1 = time.time()
    time_retro = end1 - start1
    
    del model1
    torch.cuda.empty_cache()
    time.sleep(2)
    
    # Test 2: Without RetroInfer (if context is small enough)
    if inputs['input_ids'].shape[1] < 5000:
        print("\n🔧 Without RetroInfer (Standard Attention):")
        model2 = DeepSeekModel(
            model_name=model_name,
            max_length=15000,
            dtype=torch.float16,
            device_map="auto",
            model_version="v2-lite"
        )
        
        start2 = time.time()
        out2 = model2.generate(
            attention_type="Full_Flash_Attn",
            inputs_ids=inputs["input_ids"].to(model2.layers[0].device),
            attention_masks=inputs["attention_mask"].to(model2.layers[0].device),
            max_new_length=50,
            attn_config={"Full_Flash_Attn": {}}
        )
        end2 = time.time()
        time_standard = end2 - start2
        
        # Compare results
        print(f"\n📈 Performance Comparison:")
        print(f"RetroInfer: {time_retro:.2f}s ({50/time_retro:.2f} tokens/s)")
        print(f"Standard: {time_standard:.2f}s ({50/time_standard:.2f} tokens/s)")
        print(f"Speed improvement: {(time_standard/time_retro - 1)*100:.1f}%")
    else:
        print(f"\nRetroInfer: {time_retro:.2f}s ({50/time_retro:.2f} tokens/s)")
        print("(Context too long for standard attention comparison)")
    
    return True


def test_quality_consistency():
    """Test generation quality consistency"""
    print(colored("\n✨ Test 4: Generation Quality", "yellow", attrs=["bold"]))
    
    test_prompts = [
        {
            "prompt": "请解释什么是量子计算？",
            "expected_keywords": ["量子", "叠加", "纠缠", "比特"]
        },
        {
            "prompt": "如何学习深度学习？请给出建议。",
            "expected_keywords": ["基础", "数学", "框架", "实践"]
        },
        {
            "prompt": "Python和Java的主要区别是什么？",
            "expected_keywords": ["语法", "类型", "性能", "应用"]
        }
    ]
    
    model = DeepSeekModel(
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        max_length=2048,
        dtype=torch.float16,
        device_map="auto",
        model_version="v2-lite"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")
    
    results = []
    for test in test_prompts:
        print(f"\n📝 Prompt: {test['prompt']}")
        
        inputs = tokenizer(test['prompt'], return_tensors="pt")
        outputs = model.generate(
            attention_type="RetroInfer",
            inputs_ids=inputs["input_ids"].to(model.layers[0].device),
            attention_masks=inputs["attention_mask"].to(model.layers[0].device),
            max_new_length=150,
            attn_config=get_v2_lite_config()
        )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(test['prompt']):]
        
        # Check keywords
        found_keywords = [kw for kw in test['expected_keywords'] if kw in response]
        quality_score = len(found_keywords) / len(test['expected_keywords'])
        
        print(f"Response preview: {response[:200]}...")
        print(f"Quality score: {quality_score:.2%} ({len(found_keywords)}/{len(test['expected_keywords'])} keywords)")
        
        results.append({
            "prompt": test['prompt'],
            "score": quality_score,
            "response_length": len(response)
        })
    
    # Summary
    avg_score = sum(r['score'] for r in results) / len(results)
    print(f"\n📊 Average quality score: {avg_score:.2%}")
    
    return results


def get_v2_lite_config(context_length=None):
    """Get optimized config for V2-Lite"""
    config_path = os.path.join(PROJECT_ROOT, "config/DeepSeek-V3.json")
    with open(config_path, "r") as f:
        base_config = json.load(f)
    
    v2_lite_config = base_config["DeepSeek-V2-Lite"]
    
    if context_length and context_length > 5000:
        # Adjust for longer contexts
        n_clusters = max(int(context_length / 32), 16)
        n_segments = max(int(context_length / 8192), 1)
        nprobe = max(int(n_clusters * 0.025), 4)
        
        v2_lite_config["RetroInfer"].update({
            "n_centroids": n_clusters,
            "n_segment": n_segments,
            "nprobe": nprobe
        })
    
    return {
        "RetroInfer": v2_lite_config["RetroInfer"],
        "MLA": v2_lite_config["MLA"]
    }


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-V2-Lite Specific Test Suite"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["basic", "memory", "optimization", "quality", "all"],
        default="all",
        help="Which test to run"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick version of tests"
    )
    
    args = parser.parse_args()
    
    print(colored("\n🌟 DeepSeek-V2-Lite Test Suite", "magenta", attrs=["bold"]))
    print_v2_lite_info()
    
    try:
        if args.test == "basic" or args.test == "all":
            test_basic_generation()
        
        if args.test == "memory" or args.test == "all":
            if args.quick:
                print("\n(Skipping memory test in quick mode)")
            else:
                test_memory_efficiency()
        
        if args.test == "optimization" or args.test == "all":
            test_retroinfer_optimization()
        
        if args.test == "quality" or args.test == "all":
            test_quality_consistency()
        
        print(colored("\n✅ All tests completed successfully!", "green", attrs=["bold"]))
        
    except Exception as e:
        print(colored(f"\n❌ Error: {str(e)}", "red"))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()