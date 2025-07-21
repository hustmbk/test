# deepseek_example.py - DeepSeek模型使用示例

import torch
from model_hub import DeepSeekModel
from transformers import AutoTokenizer


def test_deepseek_model():
    """
    测试DeepSeek模型的基本功能
    """
    # 模型配置
    model_name = "deepseek-ai/DeepSeek-V3"  # 实际模型路径
    max_length = 128000  # 支持128K上下文
    dtype = torch.float16
    device_map = "auto"  # 自动分配到多GPU
    
    # 初始化模型
    print("正在初始化DeepSeek模型...")
    model = DeepSeekModel(
        model_name=model_name,
        max_length=max_length,
        dtype=dtype,
        device_map=device_map,
        model_version="v3"
    )
    
    # 测试输入
    test_prompts = [
        "DeepSeek V3是一个强大的MoE模型，它的特点包括：",
        "请解释什么是Multi-head Latent Attention："
    ]
    
    # 编码输入
    tokenizer = model.tokenizer
    inputs = tokenizer(
        test_prompts,
        padding=True,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )
    
    input_ids = inputs["input_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()
    
    # 生成配置
    max_new_tokens = 512
    
    # 执行推理
    print("\n开始推理...")
    print(f"输入形状: {input_ids.shape}")
    print(f"最大新token数: {max_new_tokens}")
    
    # 使用MLA注意力机制
    outputs = model.generate(
        attention_type="MLA",  # 使用MLA注意力
        inputs_ids=input_ids,
        attention_masks=attention_mask,
        max_new_length=max_new_tokens
    )
    
    # 解码输出
    print("\n生成结果:")
    for i, output_ids in enumerate(outputs):
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        print(f"\n示例 {i+1}:")
        print(output_text)
    
    # 打印内存使用情况
    if hasattr(model.kv_cache, 'get_memory_usage'):
        memory_info = model.kv_cache.get_memory_usage()
        print(f"\n内存使用情况:")
        print(f"GPU内存: {memory_info['gpu_memory_gb']:.2f} GB")
        print(f"CPU内存: {memory_info['cpu_memory_gb']:.2f} GB")
        print(f"压缩比: {memory_info['compression_ratio']:.1f}x")


def test_moe_routing():
    """
    测试MoE路由机制
    """
    import torch.nn.functional as F
    from model_hub.moe_layer import DeepSeekMoELayer
    
    # 创建测试配置
    class MockConfig:
        num_experts = 8
        num_shared_experts = 1
        top_k = 2
        expert_hidden_dim = 2048
        hidden_size = 1024
        activation_fn = "silu"
    
    config = MockConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建MoE层
    moe_layer = DeepSeekMoELayer(0, config, device)
    
    # 创建测试输入
    batch_size = 2
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device)
    
    # 初始化路由器（简化版）
    moe_layer.router = torch.randn(config.num_experts, config.hidden_size, device=device)
    
    # 测试路由
    print("\n测试MoE路由:")
    expert_indices, expert_weights = moe_layer.route(hidden_states)
    
    print(f"专家索引形状: {expert_indices.shape}")
    print(f"专家权重形状: {expert_weights.shape}")
    print(f"选中的专家: {expert_indices[0].tolist()}")
    print(f"专家权重: {expert_weights[0].tolist()}")
    
    # 检查负载均衡
    print(f"\n专家负载分布:")
    for i in range(config.num_experts):
        load = (expert_indices == i).sum().item()
        print(f"专家 {i}: {load} 次")


def test_mla_compression():
    """
    测试MLA压缩机制
    """
    from attn_hub.mla_attn import MLAAttention
    
    # 创建测试配置
    class MockConfig:
        hidden_size = 4096
        num_heads = 32
        head_dim = 128
        kv_compress_dim = 512
        rope_dim = 64
    
    config = MockConfig()
    mla = MLAAttention(config)
    
    # 创建测试数据
    batch_size = 2
    seq_len = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device)
    W_dkv = torch.randn(config.kv_compress_dim, config.hidden_size, device=device)
    
    # 测试压缩
    print("\n测试MLA压缩:")
    print(f"原始隐藏状态形状: {hidden_states.shape}")
    
    c_kv = mla.compress_kv(hidden_states, W_dkv)
    print(f"压缩后的KV形状: {c_kv.shape}")
    print(f"压缩比: {mla.compression_ratio}x")
    
    # 计算内存节省
    original_kv_size = 2 * batch_size * seq_len * config.num_heads * config.head_dim * 2  # K和V，float16
    compressed_size = batch_size * seq_len * config.kv_compress_dim * 2  # float16
    
    print(f"\n内存使用对比:")
    print(f"标准KV缓存: {original_kv_size / (1024**2):.2f} MB")
    print(f"MLA压缩缓存: {compressed_size / (1024**2):.2f} MB")
    print(f"节省: {(1 - compressed_size/original_kv_size) * 100:.1f}%")


if __name__ == "__main__":
    print("DeepSeek模型测试")
    print("=" * 50)
    
    # 注意：这些测试需要实际的模型权重才能完全运行
    # 这里主要展示架构和API设计
    
    try:
        # 测试完整模型
        test_deepseek_model()
    except Exception as e:
        print(f"模型测试失败（预期，因为没有实际权重）: {e}")
    
    # 测试组件
    test_moe_routing()
    test_mla_compression()
    
    print("\n测试完成!")