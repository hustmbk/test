import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Attempt to import Flash Attention 2 for optimal performance
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    print("✅ Flash Attention 2 is available. Using for optimal performance.")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("⚠️ Flash Attention 2 not available. Falling back to PyTorch's scaled_dot_product_attention.")

class RMSNorm(nn.Module):
    """A PyTorch implementation of RMSNorm."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class RotaryEmbedding(nn.Module):
    """Implements the Rotary Position Embedding (RoPE)."""
    def __init__(self, dim, max_position_embeddings=4096, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Applies RoPE to query and key tensors."""
    cos = F.embedding(position_ids, cos).unsqueeze(2)
    sin = F.embedding(position_ids, sin).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DeepSeekV2Attention(nn.Module):
    """
    The core Multi-Head Latent Attention (MLA) module for DeepSeek-V2.
    This module faithfully implements the low-rank compression and attention logic.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank

        # Q-projection with LoRA
        self.q_a_proj = nn.Linear(self.hidden_size, config.q_lora_rank, bias=True)
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        # KV-projection with MLA's low-rank compression
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=True)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(self.kv_lora_rank, self.num_key_value_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        # Q projection
        q_a = self.q_a_proj(hidden_states)
        q_a = self.q_a_layernorm(q_a)
        q_b = self.q_b_proj(q_a)
        query_states = q_b.view(bsz, q_len, self.num_heads, self.q_head_dim)
        
        # KV projection (MLA)
        kv_a = self.kv_a_proj_with_mqa(hidden_states)
        kv_a_compressed = kv_a[..., :self.kv_lora_rank]
        k_pe_shared = kv_a[..., self.kv_lora_rank:]
        
        kv_a_normalized = self.kv_a_layernorm(kv_a_compressed)
        kv_b = self.kv_b_proj(kv_a_normalized)
        
        kv_b = kv_b.view(bsz, q_len, self.num_key_value_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv_b[..., :self.qk_nope_head_dim]
        value_states = kv_b[..., self.qk_nope_head_dim:]
        
        # RoPE application
        cos, sin = self.rotary_emb(value_states, seq_len=q_len + (past_key_value[0].shape[2] if past_key_value else 0))
        
        q_pe = query_states[..., self.qk_nope_head_dim:]
        q_pe_rotated, _ = apply_rotary_pos_emb(q_pe, q_pe, cos, sin, position_ids) # k is dummy here
        
        k_pe_shared = k_pe_shared.view(bsz, q_len, 1, self.qk_rope_head_dim).expand(-1, -1, self.num_key_value_heads, -1)
        _, k_pe_rotated = apply_rotary_pos_emb(k_nope, k_pe_shared, cos, sin, position_ids) # q is dummy here

        # Combine nope and rope parts
        query_states = torch.cat((query_states[..., :self.qk_nope_head_dim], q_pe_rotated), dim=-1).transpose(1, 2)
        key_states = torch.cat((k_nope, k_pe_rotated), dim=-1).transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # KV Cache management
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        # Attention computation
        if FLASH_ATTN_AVAILABLE:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, 
                causal=True,
            )
        else:
            # Fallback to PyTorch's implementation
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                is_causal=True,
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.v_head_dim)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_key_value

