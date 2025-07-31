# cache_hub/deepseek_v2_cache.py

import torch

class DeepSeekV2FlashAttnCache:
    """
    [官方做法] DeepSeek-V2 MLA 架构的专用KV缓存.

    该缓存的核心是能够存储维度不同的Key和Value。
    - Key Cache: [batch_size, max_len, num_kv_heads, key_head_dim (192)]
    - Value Cache: [batch_size, max_len, num_kv_heads, value_head_dim (128)]

    它不关心Attention的具体计算方式，只负责正确地存储和读取数据。
    """

    def __init__(
        self,
        config,
        batch_size: int,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.config = config
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.dtype = dtype

        # [官方做法] 从config中获取MLA架构的特定维度
        self.num_key_value_heads = config.num_key_value_heads
        self.key_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim  # 192
        self.value_head_dim = config.v_head_dim  # 128

        self.is_mla = True
        self.current_seq_len = 0

        # 分配内存
        self.key_cache = torch.zeros(
            (batch_size, self.max_length, self.num_key_value_heads, self.key_head_dim),
            device=self.device,
            dtype=self.dtype
        )
        self.value_cache = torch.zeros(
            (batch_size, self.max_length, self.num_key_value_heads, self.value_head_dim),
            device=self.device,
            dtype=self.dtype
        )
        print(f"✅ DeepSeekV2FlashAttnCache initialized on {self.device}.")
        print(f"   - Key Cache Shape: {self.key_cache.shape}")
        print(f"   - Value Cache Shape: {self.value_cache.shape}")
        
    def get_past_key_value(self, layer_idx: int):
        """
        [官方做法] 为HuggingFace layer的forward方法准备past_key_value。
        注意：HuggingFace期望的格式是 [bsz, num_heads, seq_len, head_dim]
        """
        if self.current_seq_len == 0:
            return None
        
        # 从我们的 [bsz, seq_len, num_heads, head_dim] 格式转换
        key = self.key_cache[:, :self.current_seq_len].transpose(1, 2)
        value = self.value_cache[:, :self.current_seq_len].transpose(1, 2)
        
        # HuggingFace的past_key_value是一个长度为num_layers的list of tuples
        # 但我们这里只为单层提供，所以直接返回元组
        return (key, value)

    def update_with_new_key_value(self, new_key_value):
        """
        [官方做法] 用HuggingFace layer返回的新KV状态来更新我们的缓存。
        
        Args:
            new_key_value (tuple): (key, value)
                - key shape: [bsz, num_kv_heads, new_total_len, k_head_dim]
                - value shape: [bsz, num_kv_heads, new_total_len, v_head_dim]
        """
        new_key, new_value = new_key_value
        new_total_len = new_key.shape[2]

        if new_total_len > self.max_length:
            raise ValueError("KV Cache overflow!")
            
        # 直接用新的完整缓存覆盖旧的
        self.key_cache[:, :new_total_len] = new_key.transpose(1, 2)
        self.value_cache[:, :new_total_len] = new_value.transpose(1, 2)
        self.current_seq_len = new_total_len

    def reset(self):
        """重置缓存"""
        self.current_seq_len = 0