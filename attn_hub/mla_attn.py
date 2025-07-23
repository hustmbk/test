# mla_attn.py - MLA（Multi-head Latent Attention）注意力机制实现
# DeepSeek V2/V3的核心创新，通过低秩压缩大幅提升效率

import torch
import torch.nn.functional as F
import flashinfer
from typing import Optional, Tuple
from utils.logger import get_logger, log_error_with_context


def apply_decoupled_rope(query_states, key_states, position_ids, rope_dim, cos_cache, sin_cache):
    """
    应用解耦的RoPE位置编码
    
    DeepSeek的创新：只对部分维度应用RoPE，其余维度保持不变
    这样可以更好地平衡位置信息和语义信息
    
    参数:
        query_states: [batch_size, seq_len, num_heads, head_dim]
        key_states: [batch_size, seq_len, num_heads, head_dim]
        position_ids: 位置索引
        rope_dim: 应用RoPE的维度数
        cos_cache, sin_cache: 预计算的cos/sin缓存
    """
    batch_size, seq_len, num_heads, head_dim = query_states.shape
    
    # 分离RoPE维度和非RoPE维度
    q_rope = query_states[..., :rope_dim]
    q_nope = query_states[..., rope_dim:]
    k_rope = key_states[..., :rope_dim]
    k_nope = key_states[..., rope_dim:]
    
    # 只对RoPE维度应用位置编码
    q_rope = q_rope.reshape(-1, rope_dim)
    k_rope = k_rope.reshape(-1, rope_dim)
    
    # 应用RoPE（使用FlashInfer的高效实现）
    flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(
        position_ids.flatten(),
        q_rope,
        k_rope,
        rope_dim,
        cos_cache,
        sin_cache,
        True  # interleaved
    )
    
    # 重塑并合并
    q_rope = q_rope.view(batch_size, seq_len, num_heads, rope_dim)
    k_rope = k_rope.view(batch_size, seq_len, num_heads, rope_dim)
    
    query_states = torch.cat([q_rope, q_nope], dim=-1)
    key_states = torch.cat([k_rope, k_nope], dim=-1)
    
    return query_states, key_states


def mla_prefill_attn(
    query_states: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_b_proj: torch.nn.Linear,
    v_b_proj: torch.nn.Linear,
    num_heads: int,
    head_dim: int,
    v_head_dim: int,
    position_ids: Optional[torch.Tensor] = None,
    rope_dim: int = 64,
    cos_cache: Optional[torch.Tensor] = None,
    sin_cache: Optional[torch.Tensor] = None,
    causal: bool = True
):
    """
    MLA预填充注意力计算
    
    核心思想：
    1. 从压缩的KV中解压得到完整的K和V
    2. 使用Flash Attention进行高效计算
    3. 支持解耦的RoPE位置编码
    
    参数:
        query_states: 查询状态 [batch_size, seq_len, num_heads * head_dim]
        compressed_kv: 压缩的KV [batch_size, seq_len, kv_lora_rank]
        k_b_proj: Key解压投影层
        v_b_proj: Value解压投影层
        num_heads: 注意力头数
        head_dim: 每个头的维度
        v_head_dim: Value的头维度（可能与Query/Key不同）
        position_ids: 位置ID
        rope_dim: RoPE应用的维度数
        cos_cache, sin_cache: RoPE缓存
        causal: 是否使用因果掩码
        
    返回:
        注意力输出 [batch_size, seq_len, num_heads * v_head_dim]
    """
    batch_size, seq_len, _ = query_states.shape
    
    # 解压缩得到K和V
    key_states = k_b_proj(compressed_kv)  # [bs, seq_len, num_heads * head_dim]
    value_states = v_b_proj(compressed_kv)  # [bs, seq_len, num_heads * v_head_dim]
    
    # 重塑为多头格式
    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim)
    key_states = key_states.view(batch_size, seq_len, num_heads, head_dim)
    value_states = value_states.view(batch_size, seq_len, num_heads, v_head_dim)
    
    # 应用解耦的RoPE
    if position_ids is not None and cos_cache is not None:
        query_states, key_states = apply_decoupled_rope(
            query_states, key_states, position_ids,
            rope_dim, cos_cache, sin_cache
        )
    
    # 使用Flash Attention进行计算
    # 转换为Flash Attention需要的格式 [bs, num_heads, seq_len, head_dim]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    # Flash Attention计算
    attn_output = flashinfer.BatchPrefillAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal
    ).forward(query_states, key_states, value_states)
    
    # 转回原始格式 [bs, seq_len, num_heads, v_head_dim]
    attn_output = attn_output.transpose(1, 2)
    
    # 合并多头 [bs, seq_len, num_heads * v_head_dim]
    attn_output = attn_output.reshape(batch_size, seq_len, -1)
    
    return attn_output


def mla_decode_attn(
    query_states: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    k_b_proj: torch.nn.Linear,
    v_b_proj: torch.nn.Linear,
    layer_idx: int,
    cache_manager,
    num_heads: int,
    head_dim: int,
    v_head_dim: int,
    position_ids: Optional[torch.Tensor] = None,
    rope_dim: int = 64,
    cos_cache: Optional[torch.Tensor] = None,
    sin_cache: Optional[torch.Tensor] = None
):
    """
    MLA解码注意力计算
    
    特点：
    1. 使用存储的压缩KV缓存
    2. 实时解压缩进行注意力计算
    3. 支持超长序列的高效推理
    
    参数:
        query_states: 新token的查询状态 [batch_size, 1, num_heads * head_dim]
        compressed_kv_cache: 历史的压缩KV缓存
        k_b_proj, v_b_proj: 解压投影层
        layer_idx: 当前层索引
        cache_manager: 缓存管理器
        其他参数同prefill
        
    返回:
        注意力输出 [batch_size, 1, num_heads * v_head_dim]
    """
    batch_size = query_states.shape[0]
    
    # 获取压缩的KV缓存
    # 这可能包括GPU和CPU部分的合并
    compressed_kv_history = cache_manager.get_compressed_kv(layer_idx)
    
    # 解压缩历史KV
    key_cache = k_b_proj(compressed_kv_history)
    value_cache = v_b_proj(compressed_kv_history)
    
    # 当前token的压缩KV
    current_compressed_kv = compressed_kv_cache
    current_key = k_b_proj(current_compressed_kv)
    current_value = v_b_proj(current_compressed_kv)
    
    # 合并历史和当前
    key_states = torch.cat([key_cache, current_key], dim=1)
    value_states = torch.cat([value_cache, current_value], dim=1)
    
    # 重塑为多头格式
    seq_len = key_states.shape[1]
    query_states = query_states.view(batch_size, 1, num_heads, head_dim)
    key_states = key_states.view(batch_size, seq_len, num_heads, head_dim)
    value_states = value_states.view(batch_size, seq_len, num_heads, v_head_dim)
    
    # 应用RoPE（只对当前查询）
    if position_ids is not None and cos_cache is not None:
        # 为了效率，可以预先计算好解码位置的RoPE
        query_states = apply_rope_to_query(
            query_states, position_ids, rope_dim, cos_cache, sin_cache
        )
    
    # 使用Flash Attention的解码kernel
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    # 解码注意力计算
    attn_output = flashinfer.BatchDecodeAttention(
        num_heads=num_heads,
        head_dim=head_dim
    ).forward(query_states, key_states, value_states)
    
    # 转回原始格式
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(batch_size, 1, -1)
    
    return attn_output


def apply_rope_to_query(query_states, position_ids, rope_dim, cos_cache, sin_cache):
    """
    仅对查询应用RoPE（解码阶段优化）
    """
    batch_size, seq_len, num_heads, head_dim = query_states.shape
    
    # 分离RoPE和非RoPE维度
    q_rope = query_states[..., :rope_dim]
    q_nope = query_states[..., rope_dim:]
    
    # 应用RoPE
    q_rope = q_rope.reshape(-1, rope_dim)
    cos = cos_cache[position_ids].reshape(-1, rope_dim)
    sin = sin_cache[position_ids].reshape(-1, rope_dim)
    
    # 旋转变换
    q_rope_real = q_rope[..., 0::2]
    q_rope_imag = q_rope[..., 1::2]
    
    q_rope_new_real = q_rope_real * cos[..., 0::2] - q_rope_imag * sin[..., 1::2]
    q_rope_new_imag = q_rope_real * sin[..., 0::2] + q_rope_imag * cos[..., 1::2]
    
    q_rope_new = torch.stack([q_rope_new_real, q_rope_new_imag], dim=-1).flatten(-2)
    
    # 重塑并合并
    q_rope_new = q_rope_new.view(batch_size, seq_len, num_heads, rope_dim)
    query_states = torch.cat([q_rope_new, q_nope], dim=-1)
    
    return query_states


class MLAAttentionWrapper:
    """
    MLA注意力的包装类，提供统一接口
    """
    
    def __init__(self, config):
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.v_head_dim = config.v_head_dim or self.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        
    def prefill(self, query_states, compressed_kv, k_b_proj, v_b_proj, 
                position_ids=None, cos_cache=None, sin_cache=None, causal=True):
        """预填充阶段的注意力计算"""
        return mla_prefill_attn(
            query_states, compressed_kv, k_b_proj, v_b_proj,
            self.num_heads, self.head_dim, self.v_head_dim,
            position_ids, self.rope_dim, cos_cache, sin_cache, causal
        )
        
    def decode(self, query_states, compressed_kv_cache, k_b_proj, v_b_proj,
               layer_idx, cache_manager, position_ids=None, 
               cos_cache=None, sin_cache=None):
        """解码阶段的注意力计算"""
        return mla_decode_attn(
            query_states, compressed_kv_cache, k_b_proj, v_b_proj,
            layer_idx, cache_manager, self.num_heads, self.head_dim,
            self.v_head_dim, position_ids, self.rope_dim, cos_cache, sin_cache
        )