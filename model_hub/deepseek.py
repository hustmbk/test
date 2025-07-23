# deepseek.py - DeepSeek V2/V3 MoE模型实现
# 支持MLA（Multi-head Latent Attention）和稀疏MoE架构
# 实现了高效的KV缓存压缩和专家路由机制

import gc
import re
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import flashinfer
from transformers import AutoTokenizer, AutoConfig
from .LLM import LLM
from cache_hub import mla_cache, flash_attn_cache
from attn_hub import mla_prefill_attn, mla_decode_attn
import math
from utils.logger import get_logger, log_error_with_context, log_gpu_memory


def safe_int_config(config, attr_name, default_value):
    """
    安全地从配置中获取整数值
    
    处理配置值为None、字符串或其他类型的情况
    """
    try:
        value = getattr(config, attr_name, default_value)
        if value is None:
            return int(default_value)
        return int(value)
    except (ValueError, TypeError) as e:
        logger = get_logger()
        logger.warning(f"配置项 {attr_name} 值无效: {value}, 使用默认值: {default_value}")
        return int(default_value)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    DeepSeek和Llama使用的归一化方法
    相比LayerNorm，RMSNorm计算更高效
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MLAAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) 实现
    
    DeepSeek V2的核心创新：通过低秩压缩大幅减少KV缓存
    - 将KV投影到低维潜在空间（压缩比高达32倍）
    - 解耦的RoPE位置编码
    - 支持长序列高效推理
    """
    
    def __init__(self, config, layer_idx, device):
        super().__init__()
        self.layer_idx = layer_idx
        self.device = device
        
        logger = get_logger()
        logger.debug(f"初始化MLAAttention层 {layer_idx}", device=device)
        
        # 验证必要的配置参数
        if not hasattr(config, 'hidden_size'):
            logger.error(f"MLAAttention层 {layer_idx} 初始化失败: 配置中缺少 'hidden_size' 属性")
            raise ValueError("Config must have 'hidden_size' attribute")
        if not hasattr(config, 'num_attention_heads'):
            logger.error(f"MLAAttention层 {layer_idx} 初始化失败: 配置中缺少 'num_attention_heads' 属性")
            raise ValueError("Config must have 'num_attention_heads' attribute")
        
        # MLA配置参数
        self.hidden_size = safe_int_config(config, 'hidden_size', 4096)
        self.num_heads = safe_int_config(config, 'num_attention_heads', 32)
        self.head_dim = int(self.hidden_size // self.num_heads)
        
        # 验证head_dim
        if self.hidden_size % self.num_heads != 0:
            logger.error(f"MLAAttention层 {layer_idx} 初始化失败: hidden_size ({self.hidden_size}) 不能被 num_heads ({self.num_heads}) 整除")
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})")
        
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        # MLA特定参数
        self.q_lora_rank = safe_int_config(config, 'q_lora_rank', 1536)  # Query LoRA秩
        self.kv_lora_rank = safe_int_config(config, 'kv_lora_rank', 512)  # 压缩的KV维度（如512）
        self.qk_rope_head_dim = safe_int_config(config, 'qk_rope_head_dim', 64)  # RoPE维度
        self.qk_nope_head_dim = int(self.head_dim - self.qk_rope_head_dim)  # 非RoPE维度
        self.v_head_dim = safe_int_config(config, 'v_head_dim', self.head_dim)  # V维度
        
        logger.info(f"MLAAttention层 {layer_idx} 配置完成", 
                   hidden_size=self.hidden_size,
                   num_heads=self.num_heads, 
                   head_dim=self.head_dim,
                   q_lora_rank=self.q_lora_rank,
                   kv_lora_rank=self.kv_lora_rank,
                   compression_ratio=f"{self.hidden_size/self.kv_lora_rank:.1f}x")
        
        # 初始化MLA投影矩阵
        self._init_mla_projections()
        logger.debug(f"MLAAttention层 {layer_idx} 投影矩阵初始化完成")
        
    def _init_mla_projections(self):
        """初始化MLA的低秩投影矩阵"""
        logger = get_logger()
        
        try:
            # 确保所有维度都是整数
            hidden_size = int(self.hidden_size)
            q_lora_rank = int(self.q_lora_rank)
            kv_lora_rank = int(self.kv_lora_rank)
            num_heads = int(self.num_heads)
            head_dim = int(self.head_dim)
            v_head_dim = int(self.v_head_dim)
            
            logger.debug(f"MLA投影参数: hidden_size={hidden_size}, q_lora_rank={q_lora_rank}, "
                        f"kv_lora_rank={kv_lora_rank}, num_heads={num_heads}, "
                        f"head_dim={head_dim}, v_head_dim={v_head_dim}")
            
            # 验证参数合理性
            if hidden_size <= 0 or q_lora_rank <= 0 or kv_lora_rank <= 0:
                raise ValueError(f"Invalid dimensions: hidden_size={hidden_size}, "
                               f"q_lora_rank={q_lora_rank}, kv_lora_rank={kv_lora_rank}")
            
            # Query低秩分解: hidden -> q_lora_rank -> q_heads * head_dim
            self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
            self.q_b_proj = nn.Linear(q_lora_rank, num_heads * head_dim, bias=False)
            
            # 压缩的KV投影: hidden -> kv_lora_rank
            self.kv_a_proj = nn.Linear(hidden_size, kv_lora_rank, bias=False)
            
            # 解压缩投影
            # Key: kv_lora_rank -> num_heads * (qk_nope_head_dim + qk_rope_head_dim)
            self.k_b_proj = nn.Linear(kv_lora_rank, num_heads * head_dim, bias=False)
            
            # Value: kv_lora_rank -> num_heads * v_head_dim
            self.v_b_proj = nn.Linear(kv_lora_rank, num_heads * v_head_dim, bias=False)
            
            # 输出投影
            self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)
            
            logger.debug("MLA投影矩阵创建成功")
            
        except Exception as e:
            logger.error(f"MLA投影初始化失败: {str(e)}", 
                        hidden_size=getattr(self, 'hidden_size', 'undefined'),
                        q_lora_rank=getattr(self, 'q_lora_rank', 'undefined'),
                        kv_lora_rank=getattr(self, 'kv_lora_rank', 'undefined'),
                        num_heads=getattr(self, 'num_heads', 'undefined'),
                        head_dim=getattr(self, 'head_dim', 'undefined'))
            raise
        
    def forward(self, hidden_states, position_ids=None, attention_mask=None, use_cache=True):
        """
        MLA前向传播
        
        参数:
            hidden_states: 输入隐藏状态 [batch_size, seq_len, hidden_size]
            position_ids: 位置ID
            attention_mask: 注意力掩码
            use_cache: 是否使用KV缓存
            
        返回:
            attn_output: 注意力输出
            compressed_kv: 压缩的KV状态（用于缓存）
        """
        logger = get_logger()
        batch_size, seq_len, _ = hidden_states.shape
        
        logger.debug(f"MLAAttention层 {self.layer_idx} 前向传播开始", 
                    batch_size=batch_size, seq_len=seq_len, use_cache=use_cache)
        
        try:
            # Query低秩投影
            q_compressed = self.q_a_proj(hidden_states)  # [bs, seq_len, q_lora_rank]
            query_states = self.q_b_proj(q_compressed).to(self.parent_model.dtype)  # 强制转换为模型数据类型
            
            # KV压缩投影（这是MLA的核心优化）
            compressed_kv = self.kv_a_proj(hidden_states)  # [bs, seq_len, kv_lora_rank]
            
            # 解压缩得到K和V，并强制转换数据类型
            key_states = self.k_b_proj(compressed_kv).to(self.parent_model.dtype)  # 强制转换为模型数据类型
            value_states = self.v_b_proj(compressed_kv).to(self.parent_model.dtype)  # 强制转换为模型数据类型
            
            # 重塑为多头格式
            query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
            key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
            value_states = value_states.view(batch_size, seq_len, self.num_heads, self.v_head_dim)
            
            # 应用RoPE（仅对部分维度）
            if position_ids is not None:
                query_states, key_states = self.apply_rope(query_states, key_states, position_ids)
            
            # 注意力计算
            attn_output = self.compute_attention(query_states, key_states, value_states, attention_mask)
            
            # 输出投影
            attn_output = attn_output.reshape(batch_size, seq_len, -1)
            attn_output = self.o_proj(attn_output)
            
            logger.debug(f"MLAAttention层 {self.layer_idx} 前向传播完成", 
                        output_shape=attn_output.shape)
            
            return attn_output, compressed_kv if use_cache else None
            
        except Exception as e:
            log_error_with_context(e, {
                "layer_idx": self.layer_idx,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "input_shape": hidden_states.shape,
                "use_cache": use_cache
            })
            raise
    
    def apply_rope(self, query_states, key_states, position_ids):
        """
        应用解耦的RoPE位置编码
        只对qk_rope_head_dim维度应用RoPE，其余维度保持不变
        """
        # 使用父模型的合并缓存
        if hasattr(self, 'parent_model') and hasattr(self.parent_model, 'cos_sin_cache_merged'):
            cos_sin_cache_merged = self.parent_model.cos_sin_cache_merged
            
            # 导入解耦RoPE函数
            from attn_hub.mla_attn import apply_decoupled_rope
            
            return apply_decoupled_rope(
                query_states, key_states, position_ids,
                self.qk_rope_head_dim, cos_sin_cache_merged, None  # 传递合并缓存，sin_cache设为None
            )
        else:
            # 如果没有缓存，返回原始状态
            return query_states, key_states
    
    def compute_attention(self, query_states, key_states, value_states, attention_mask):
        """计算注意力（使用Flash Attention优化）"""
        batch_size, seq_len, num_heads, head_dim = query_states.shape
        
        # 转换为Flash Attention需要的格式 [bs, num_heads, seq_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # 使用FlashInfer进行高效注意力计算
        attn_output = flashinfer.BatchPrefillAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            causal=True  # 使用因果掩码
        ).forward(query_states, key_states, value_states)
        
        # 转回原始格式 [bs, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2)
        
        # 合并多头
        
        return attn_output
    
    def decode_with_compressed_kv(self, hidden_states, compressed_kv_cache, position_ids=None):
        """
        使用压缩的KV缓存进行解码
        
        参数:
            hidden_states: 新token的隐藏状态 [batch_size, 1, hidden_size]
            compressed_kv_cache: 从缓存管理器获取的压缩KV
            position_ids: 位置ID
            
        返回:
            注意力输出 [batch_size, 1, hidden_size]
        """
        batch_size = hidden_states.shape[0]
        
        # Query低秩投影
        q_compressed = self.q_a_proj(hidden_states)
        query_states = self.q_b_proj(q_compressed).to(self.parent_model.dtype)  # 强制转换为模型数据类型
        
        # 重塑为多头格式
        query_states = query_states.view(batch_size, 1, self.num_heads, self.head_dim)
        
        # 使用mla_decode_attn进行解码
        if hasattr(self, 'parent_model') and hasattr(self.parent_model, 'cos_sin_cache'):
            cos_cache, sin_cache = self.parent_model.cos_sin_cache
        else:
            cos_cache, sin_cache = None, None
            
        # 调用解码注意力函数
        attn_output = mla_decode_attn(
            query_states.view(batch_size, 1, -1),  # 展平多头
            compressed_kv_cache,
            self.k_b_proj,
            self.v_b_proj,
            self.layer_idx,
            self.parent_model.kv_cache if hasattr(self, 'parent_model') else None,
            self.num_heads,
            self.head_dim,
            self.v_head_dim,
            position_ids,
            self.qk_rope_head_dim,
            cos_cache,
            sin_cache
        )
        
        # 输出投影
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class DeepSeekMoELayer(nn.Module):
    """
    DeepSeek MoE层实现
    
    特点：
    - 细粒度专家：大量小专家（如256个）
    - 共享专家：所有token都会经过的专家
    - 动态路由：每个token选择Top-K个专家
    - 无辅助损失的负载均衡
    """
    
    def __init__(self, config, layer_idx, device):
        super().__init__()
        self.layer_idx = layer_idx
        self.device = device
        
        logger = get_logger()
        logger.debug(f"初始化DeepSeekMoELayer层 {layer_idx}", device=device)
        
        # MoE配置
        self.hidden_size = safe_int_config(config, 'hidden_size', 4096)
        self.intermediate_size = safe_int_config(config, 'intermediate_size', 11008)
        self.num_experts = safe_int_config(config, 'num_experts', 256)  # 路由专家数量（如256）
        self.num_experts_per_tok = safe_int_config(config, 'num_experts_per_tok', 8)  # 每个token激活的专家数（如8）
        self.moe_intermediate_size = safe_int_config(config, 'moe_intermediate_size', self.intermediate_size // 8)  # 每个专家的中间层大小
        
        logger.info(f"MoE层 {layer_idx} 配置", 
                   num_experts=self.num_experts,
                   experts_per_token=self.num_experts_per_tok,
                   expert_size=self.moe_intermediate_size,
                   activation_ratio=f"{self.num_experts_per_tok/self.num_experts*100:.1f}%")
        
        # 共享专家（所有token都经过）
        self.shared_expert = nn.Sequential(
            nn.Linear(self.hidden_size, self.intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        )
        
        # 路由器
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        
        # 专家网络（细粒度专家）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.moe_intermediate_size, bias=False),
                nn.SiLU(),
                nn.Linear(self.moe_intermediate_size, self.hidden_size, bias=False)
            ) for _ in range(self.num_experts)
        ])
        
        logger.debug(f"MoE层 {layer_idx} 初始化完成: {self.num_experts}个专家 + 1个共享专家")
        
    def forward(self, hidden_states):
        """
        MoE前向传播
        
        参数:
            hidden_states: 输入隐藏状态 [batch_size, seq_len, hidden_size]
            
        返回:
            output: MoE输出
            router_logits: 路由器logits（用于负载均衡分析）
        """
        if hidden_states.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got {hidden_states.dim()}D")
            
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if hidden_dim != self.hidden_size:
            raise ValueError(f"Input hidden dimension ({hidden_dim}) doesn't match expected ({self.hidden_size})")
        
        # 共享专家的输出
        shared_output = self.shared_expert(hidden_states)
        
        # 计算路由分数
        router_logits = self.gate(hidden_states)  # [bs, seq_len, num_experts]
        
        # 选择Top-K专家
        routing_weights, selected_experts = torch.topk(
            router_logits, self.num_experts_per_tok, dim=-1
        )
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # 初始化专家输出
        expert_output = torch.zeros_like(hidden_states)
        
        # 对每个专家进行计算（实际实现中需要优化）
        for expert_idx in range(self.num_experts):
            # 找出选择了这个专家的token
            expert_mask = (selected_experts == expert_idx).any(dim=-1)
            if expert_mask.any():
                # 获取该专家的输入
                expert_input = hidden_states[expert_mask]
                # 计算专家输出
                expert_out = self.experts[expert_idx](expert_input)
                # 获取对应的权重
                expert_weights = routing_weights[expert_mask]
                expert_weights = expert_weights[selected_experts[expert_mask] == expert_idx]
                # 加权输出
                expert_output[expert_mask] += expert_weights.unsqueeze(-1) * expert_out
        
        # 合并共享专家和路由专家的输出
        output = shared_output + expert_output
        
        return output, router_logits


class DeepSeekLayer(nn.Module):
    """
    DeepSeek单层封装
    
    集成了MLA注意力和MoE FFN
    """
    
    def __init__(self, config, layer_idx, device):
        super().__init__()
        self.layer_idx = layer_idx
        self.device = device
        
        # 初始化MLA注意力
        self.self_attn = MLAAttention(config, layer_idx, device).to(device)
        
        # 初始化MoE FFN
        self.mlp = DeepSeekMoELayer(config, layer_idx, device).to(device)
        
        # LayerNorm (使用自定义RMSNorm)
        rms_norm_eps = getattr(config, 'rms_norm_eps', 1e-6)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=rms_norm_eps).to(device)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=rms_norm_eps).to(device)
        
    def init_from_hf(self, hf_layer):
        """从HuggingFace模型初始化权重"""
        # TODO: 实现权重转换
        pass


class DeepSeekModel(LLM):
    """
    DeepSeek V2/V3模型实现
    
    主要特性：
    1. MLA注意力：93.3%的KV缓存压缩
    2. MoE架构：仅激活5.5%的参数
    3. 支持128K+上下文长度
    4. 多GPU并行推理
    """
    
    def __init__(
        self, 
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str,
        model_version: str = "v3"  # "v2", "v2-lite", or "v3"
    ) -> None:
        """
        初始化DeepSeek模型
        
        参数:
            model_name: 模型名称或路径
            max_length: 最大序列长度
            dtype: 数据类型
            device_map: 设备映射策略
            model_version: 模型版本（v2, v2-lite, 或 v3）
        """
        logger = get_logger()
        logger.info(f"开始初始化DeepSeek {model_version}模型", 
                   model_name=model_name, 
                   max_length=max_length,
                   dtype=str(dtype),
                   device_map=device_map)
        
        super().__init__(model_name, max_length, dtype, device_map)
        
        self.model_version = model_version
        self.attention_type = 'MLA'  # DeepSeek使用MLA注意力
        
        # 加载配置和tokenizer
        # 检查是否是本地路径
        import os
        local_files_only = os.path.exists(model_name) and os.path.isdir(model_name)
        
        logger.info(f"加载模型配置和tokenizer", local_files_only=local_files_only)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
            self.config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
            logger.info("成功加载tokenizer和配置")
        except Exception as e:
            logger.error(f"加载模型失败: {model_name}", error=e)
            raise ValueError(f"Failed to load model from {model_name}: {str(e)}")
        
        # 验证必要的配置属性
        required_attrs = ['num_hidden_layers', 'hidden_size', 'vocab_size']
        for attr in required_attrs:
            if not hasattr(self.config, attr):
                logger.error(f"模型配置缺少必要属性: {attr}")
                raise ValueError(f"Model config missing required attribute: {attr}")
        
        # 提取模型参数
        self.num_layers = safe_int_config(self.config, 'num_hidden_layers', 32)
        self.hidden_size = safe_int_config(self.config, 'hidden_size', 4096)
        self.vocab_size = safe_int_config(self.config, 'vocab_size', 50257)
        
        # MLA特定参数 (可能不存在于所有配置中)
        self.kv_lora_rank = safe_int_config(self.config, 'kv_lora_rank', 512)  # 默认值512
        self.q_lora_rank = safe_int_config(self.config, 'q_lora_rank', 1536)  # 默认值1536
        
        # MoE特定参数 (可能不存在于所有配置中)
        self.num_experts = safe_int_config(self.config, 'num_experts', 0)  # 0表示不使用MoE
        self.num_experts_per_tok = safe_int_config(self.config, 'num_experts_per_tok', 0)
        
        # 根据模型版本设置默认值
        if self.num_experts == 0:
            if model_version == "v3":
                self.num_experts = 256
                self.num_experts_per_tok = 8
            elif model_version == "v2":
                self.num_experts = 160
                self.num_experts_per_tok = 6
            elif model_version == "v2-lite":
                self.num_experts = 64
                self.num_experts_per_tok = 6
        
        # 设置模型大小（用于内存估算）
        if model_version == "v3":
            self.model_size_gb = 336  # DeepSeek V3 约336GB
        elif model_version == "v2":
            self.model_size_gb = 118  # DeepSeek V2 约118GB  
        elif model_version == "v2-lite":
            self.model_size_gb = 8   # DeepSeek V2 Lite 约8GB
        else:
            self.model_size_gb = 8   # 默认值
        
        logger.info(f"DeepSeek {model_version} 模型参数", 
                   num_layers=self.num_layers,
                   hidden_size=self.hidden_size,
                   vocab_size=self.vocab_size,
                   kv_compression=f"{self.hidden_size/self.kv_lora_rank:.1f}x",
                   num_experts=self.num_experts,
                   experts_per_token=self.num_experts_per_tok)
        
        # 初始化模型
        log_gpu_memory(phase="before_model_init")
        self.init_model()
        log_gpu_memory(phase="after_model_init")
        
        logger.info(f"DeepSeek {model_version} 模型初始化完成")
        
    def init_model(self):
        """初始化模型权重和结构"""
        # 确定GPU数量和设备映射
        self.num_gpus = torch.cuda.device_count() if self.device_map == 'auto' else 1
        if self.device_map == 'auto' and self.num_gpus == 1:
            self.device_map = 'cuda:0'
            
        # 设置模型级别的注意力参数（用于缓存初始化）
        self.num_heads = safe_int_config(self.config, 'num_attention_heads', 32)
        self.num_key_value_heads = safe_int_config(self.config, 'num_key_value_heads', self.num_heads)
        self.head_dim = int(self.hidden_size // self.num_heads)
        
        # 创建层到设备的映射
        if self.device_map != "auto":  # 单GPU
            self.layer_mapping = {str(i): self.device_map for i in range(self.num_layers)}
        else:  # 多GPU
            self.gpu_ids = list(range(self.num_gpus))
            self.layer_interval = (self.num_layers + self.num_gpus - 1) // self.num_gpus
            self.layer_mapping = {
                str(i): f'cuda:{i // self.layer_interval}' 
                for i in range(self.num_layers)
            }
            
        # 初始化embedding和输出层
        device_0 = self.device_map if self.device_map != "auto" else f'cuda:{self.gpu_ids[0]}'
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size).to(device_0)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False).to(device_0)
        
        # 初始化最终的RMSNorm
        rms_norm_eps = getattr(self.config, 'rms_norm_eps', 1e-6)
        self.norm = RMSNorm(self.hidden_size, eps=rms_norm_eps).to(device_0)
        
        # 初始化位置编码 - 确保使用正确的数据类型
        self.position_ids = torch.arange(0, self.max_length, dtype=torch.int32).to(device_0)
        self._init_rope()
        
        # 初始化各层
        self.layers = []
        for idx in range(self.num_layers):
            device = self.layer_mapping[str(idx)]
            layer = DeepSeekLayer(self.config, idx, device)
            self.layers.append(layer)
            
        # 为每个注意力层设置父模型引用（用于访问cos_sin_cache）
        for layer in self.layers:
            if hasattr(layer, 'self_attn'):
                layer.self_attn.parent_model = self
            
        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()
        
    def _init_rope(self):
        """初始化RoPE位置编码"""
        # 获取RoPE参数
        rope_theta = getattr(self.config, 'rope_theta', 10000.0)
        rope_dim = getattr(self.config, 'qk_rope_head_dim', 64)
        
        # 确保rope_dim是偶数且不为0
        if rope_dim <= 0 or rope_dim % 2 != 0:
            rope_dim = 64  # 使用默认值
            
        # 预计算cos和sin缓存 - 使用模型数据类型而不是float32
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        t = torch.arange(self.max_length, dtype=self.dtype)  # 使用模型数据类型
        freqs = torch.einsum('i,j->ij', t, inv_freq.to(self.dtype))  # 确保数据类型一致
        
        # 创建cos和sin缓存 - 使用模型数据类型（通常是fp16）
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        # 安全地扩展维度以匹配rope_dim
        if rope_dim > 0 and cos.numel() > 0 and sin.numel() > 0:
            # 确保reshape参数正确
            cos_expanded = torch.stack([cos, cos], dim=-1)
            sin_expanded = torch.stack([sin, sin], dim=-1)
            cos_cache = cos_expanded.reshape(self.max_length, rope_dim)
            sin_cache = sin_expanded.reshape(self.max_length, rope_dim)
        else:
            # 如果遇到空张量，创建默认的缓存（使用模型数据类型）
            cos_cache = torch.zeros((self.max_length, rope_dim), dtype=self.dtype)
            sin_cache = torch.zeros((self.max_length, rope_dim), dtype=self.dtype)
        
        # 移动到设备并转换为模型相同的数据类型（FlashInfer要求fp16/bf16，不支持float32）
        device_0 = self.device_map if self.device_map != "auto" else f'cuda:{self.gpu_ids[0]}'
        model_dtype = self.dtype  # 使用模型的数据类型（通常是torch.float16）
        cos_cache = cos_cache.to(device_0, dtype=model_dtype)  # 改为模型数据类型
        sin_cache = sin_cache.to(device_0, dtype=model_dtype)  # 改为模型数据类型
        
        # 存储分别的cos和sin缓存（用于MLA）
        self.cos_sin_cache = (cos_cache, sin_cache)
        
        # 同时创建合并的缓存（用于FlashInfer）- 使用模型数据类型
        self.cos_sin_cache_merged = torch.cat([cos_cache, sin_cache], dim=-1)
        
    def init_kv_cache(self, real_input_length, valid_start, attn_config=None):
        """
        初始化MLA压缩KV缓存
        
        与标准KV缓存的主要区别：
        - 存储压缩的潜在向量而非完整的K和V
        - 大幅减少内存使用（压缩比32倍）
        """
        logger = get_logger()
        logger.info("初始化MLA压缩KV缓存", 
                   input_length=real_input_length,
                   batch_size=self.batch_size,
                   max_new_length=self.max_new_length,
                   kv_lora_rank=self.kv_lora_rank)
        
        if not hasattr(self, 'attention_type'):
            self.attention_type = 'MLA'
            
        if self.attention_type == 'MLA':
            try:
                log_gpu_memory(phase="before_mla_cache_init")
                self.kv_cache = mla_cache(
                    valid_start=valid_start,
                    layer_num=self.num_layers,
                    batch_size=self.batch_size,
                    max_length=self.max_new_length + real_input_length,
                    kv_lora_rank=self.kv_lora_rank,  # 压缩维度
                    dtype=self.dtype,
                    layer_mapping=self.layer_mapping,
                    num_gpus=self.num_gpus,
                    model_version=self.model_version
                )
                log_gpu_memory(phase="after_mla_cache_init")
                logger.info(f"MLA缓存初始化成功，压缩比: {self.hidden_size/self.kv_lora_rank:.1f}x")
            except Exception as e:
                logger.warning(f"MLA缓存初始化失败: {str(e)}")
                logger.info("回退到标准Flash Attention缓存")
                self.attention_type = 'Flash_Attention'
                super().init_kv_cache(real_input_length, valid_start, attn_config)
        else:
            # 回退到标准Flash Attention缓存
            logger.info("使用标准Flash Attention缓存")
            super().init_kv_cache(real_input_length, valid_start, attn_config)
            
    def word_embedding(self, input_ids):
        """词嵌入"""
        return self.embed_tokens(input_ids)
        
    def lm(self, hidden_states):
        """语言模型头"""
        return self.lm_head(hidden_states).float()
        
    def layernorm(self, hidden_states, layer_norm):
        """RMSNorm归一化"""
        return layer_norm(hidden_states)
        
    def layer_prefill(self, layer_idx, start_bdx, hidden_states):
        """
        预填充阶段的单层处理（覆盖基类方法）
        
        主要区别：
        1. 使用MLA注意力而非标准注意力
        2. 使用MoE FFN而非标准FFN
        3. 存储压缩的KV而非完整KV
        """
        logger = get_logger()
        
        if layer_idx >= len(self.layers):
            logger.error(f"层索引超出范围: {layer_idx} >= {len(self.layers)}")
            raise ValueError(f"Layer index {layer_idx} out of range (0-{len(self.layers)-1})")
            
        logger.debug(f"开始处理第{layer_idx}层预填充", 
                    start_bdx=start_bdx, 
                    input_shape=hidden_states.shape)
        
        try:
            layer = self.layers[layer_idx]
            bsz, seq_len, _ = hidden_states.shape
            
            # Pre-norm
            residual = hidden_states
            hidden_states = self.layernorm(hidden_states, layer.input_layernorm)
            
            # MLA注意力
            attn_output, compressed_kv = layer.self_attn(
                hidden_states,
                position_ids=self.position_ids[self.kv_cache.context:self.kv_cache.context+seq_len].unsqueeze(0).repeat(bsz, 1),
                use_cache=True
            )
            
            # 更新压缩的KV缓存
            if compressed_kv is not None:
                self.kv_cache.update_compressed_kv(compressed_kv, layer_idx, start_bdx)
                logger.debug(f"第{layer_idx}层压缩KV缓存已更新", 
                            compressed_kv_shape=compressed_kv.shape)
                
            hidden_states = residual + attn_output
            
            # Post-norm和MoE FFN
            residual = hidden_states
            hidden_states = self.layernorm(hidden_states, layer.post_attention_layernorm)
            moe_output, _ = layer.mlp(hidden_states)  # 忽略router_logits
            hidden_states = residual + moe_output
            
            logger.debug(f"第{layer_idx}层预填充处理完成", output_shape=hidden_states.shape)
            
            return hidden_states
            
        except Exception as e:
            log_error_with_context(e, {
                "layer_idx": layer_idx,
                "start_bdx": start_bdx,
                "input_shape": hidden_states.shape if 'hidden_states' in locals() else "unknown",
                "phase": "prefill"
            })
            raise
        
    def layer_decode(self, layer_idx, hidden_states):
        """
        解码阶段的单层处理（覆盖基类方法）
        
        使用存储的压缩KV进行高效解码
        """
        layer = self.layers[layer_idx]
        bsz, seq_len, _ = hidden_states.shape
        
        # Pre-norm
        residual = hidden_states
        hidden_states = self.layernorm(hidden_states, layer.input_layernorm)
        
        # 从缓存获取压缩的KV并进行MLA注意力计算
        compressed_kv = self.kv_cache.get_compressed_kv(layer_idx)
        attn_output = layer.self_attn.decode_with_compressed_kv(
            hidden_states,
            compressed_kv,
            position_ids=self.position_ids[self.kv_cache.context:self.kv_cache.context+seq_len]
        )
        
        hidden_states = residual + attn_output
        
        # Post-norm和MoE FFN
        residual = hidden_states
        hidden_states = self.layernorm(hidden_states, layer.post_attention_layernorm)
        moe_output, _ = layer.mlp(hidden_states)
        hidden_states = residual + moe_output
        
        return hidden_states
        
    def get_model_info(self):
        """获取模型信息"""
        # 根据版本获取参数
        if self.model_version == "v3":
            total_params = 671e9
            active_params = 37e9
        elif self.model_version == "v2":
            total_params = 236e9
            active_params = 21e9
        elif self.model_version == "v2-lite":
            total_params = 15.7e9
            active_params = 2.8e9
        else:
            # 默认值
            total_params = sum(p.numel() for p in self.parameters())
            active_params = self.hidden_size * 2
        
        info = {
            "model_version": self.model_version,
            "total_parameters": f"{total_params/1e9:.1f}B",
            "active_parameters": f"{active_params/1e9:.1f}B",
            "activation_ratio": f"{(active_params/total_params)*100:.1f}%",
            "num_experts": self.num_experts,
            "experts_per_token": self.num_experts_per_tok,
            "kv_compression_ratio": f"{self.hidden_size/self.kv_lora_rank:.1f}x",
            "max_context_length": self.max_length
        }
        
        return info