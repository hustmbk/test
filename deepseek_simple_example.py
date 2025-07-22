#!/usr/bin/env python3
"""
Simple DeepSeek-V3 + RetroInfer Example

This script demonstrates basic usage of DeepSeek-V3 with RetroInfer optimization
for efficient long-context generation.
"""

import os
import sys
import torch
import json
from termcolor import colored
from transformers import AutoTokenizer

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from model_hub import DeepSeekModel


def main():
    # Model configuration
    model_name = "deepseek-ai/DeepSeek-V3"
    model_version = "v3"  # Can be "v2", "v2-lite", or "v3"
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="DeepSeek + RetroInfer Example")
    parser.add_argument("--model-version", type=str, default="v3", 
                        choices=["v2", "v2-lite", "v3"],
                        help="Model version to use")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Local model path (overrides default model name)")
    args = parser.parse_args()
    model_version = args.model_version
    
    # Use local path if provided
    if args.model_path:
        model_name = args.model_path
        print(f"Using local model path: {model_name}")
    
    print(colored(f"DeepSeek-{model_version.upper()} + RetroInfer Simple Example", "cyan", attrs=["bold"]))
    print("=" * 50)
    
    # Example input
    input_text = """
    深度学习技术的快速发展推动了人工智能的革命性进步。特别是大语言模型(LLM)的出现，
    彻底改变了自然语言处理的格局。然而，随着模型规模的增长，如何高效地进行推理成为关键挑战。
    
    DeepSeek-V3通过创新的MLA（Multi-head Latent Attention）机制，将KV缓存压缩了32倍，
    同时采用细粒度的MoE架构，拥有671B参数但每次仅激活37B（5.5%）。
    
    RetroInfer进一步优化了长序列处理，通过将KV缓存视为向量数据库，实现了GPU和CPU的协同计算。
    
    请详细解释MLA机制是如何实现32倍压缩的？
    """
    
    print(f"Input text preview: {input_text[:200]}...")
    
    # Initialize tokenizer
    print(colored("\n1. Initializing tokenizer...", "yellow"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    print(f"✓ Input length: {input_ids.shape[1]} tokens")
    
    # Initialize model
    print(colored(f"\n2. Loading DeepSeek-{model_version.upper()} model...", "yellow"))
    model = DeepSeekModel(
        model_name=model_name,
        max_length=150000,  # Support up to 150K tokens
        dtype=torch.float16,
        device_map="auto",
        model_version=model_version
    )
    print("✓ Model loaded successfully")
    
    # Generate RetroInfer configuration
    print(colored("\n3. Configuring RetroInfer...", "yellow"))
    config_path = os.path.join(PROJECT_ROOT, "config/DeepSeek-V3.json")
    with open(config_path, "r") as f:
        base_config = json.load(f)
    
    # Get version-specific config
    if model_version == "v3":
        version_key = "DeepSeek-V3"
    elif model_version == "v2":
        version_key = "DeepSeek-V2"
    elif model_version == "v2-lite":
        version_key = "DeepSeek-V2-Lite"
    else:
        version_key = "DeepSeek-V3"
    
    # Use default RetroInfer settings for this context length
    retroinfer_config = base_config[version_key]["RetroInfer"].copy()
    
    # For longer contexts, adjust parameters dynamically
    context_len = input_ids.shape[1]
    if context_len > 10000:
        n_clusters = max(int(context_len / 16), 32)
        n_segments = max(int(context_len / 8192), 1)
        retroinfer_config.update({
            "n_centroids": n_clusters,
            "n_segment": n_segments,
            "nprobe": max(int(n_clusters * 0.018), 8)
        })
    
    attn_config = {
        "RetroInfer": retroinfer_config,
        "MLA": base_config[version_key]["MLA"]
    }
    
    print(f"✓ RetroInfer configured:")
    print(f"  - Centroids: {retroinfer_config['n_centroids']}")
    print(f"  - Segments: {retroinfer_config['n_segment']}")
    print(f"  - Probes: {retroinfer_config['nprobe']}")
    
    # Generate
    print(colored("\n4. Generating response...", "yellow"))
    print("This may take a moment...")
    
    import time
    start_time = time.time()
    
    outputs = model.generate(
        attention_type="RetroInfer",
        inputs_ids=input_ids.to(model.layers[0].device),
        attention_masks=attention_mask.to(model.layers[0].device),
        max_new_length=200,  # Generate 200 tokens
        attn_config=attn_config
    )
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    # Decode output
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Extract only the generated part
    generated_part = generated_text[len(input_text):]
    
    # Display results
    print(colored("\n5. Results:", "green"))
    print("=" * 50)
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Speed: {200/generation_time:.2f} tokens/second")
    print(f"\nGenerated response:")
    print("-" * 50)
    print(generated_part)
    print("-" * 50)
    
    # Show model info
    model_info = model.get_model_info()
    print(colored("\nModel Information:", "cyan"))
    for key, value in model_info.items():
        print(f"  • {key}: {value}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(colored(f"\nError: {str(e)}", "red"))
        import traceback
        traceback.print_exc()