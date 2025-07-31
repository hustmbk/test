import torch
import torch.nn as nn

from .deepseek_v2_attention import DeepSeekV2Attention, RMSNorm
from .deepseek_v2_mlp import DeepSeekV2MoE, DeepSeekV2MLP

class DeepSeekV2DecoderLayer(nn.Module):
    """
    A complete Transformer Decoder Layer for DeepSeek-V2.
    It combines the MLA attention and the MLP/MoE blocks.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = DeepSeekV2Attention(config)
        
        # Determine if this layer should be MoE or dense
        is_moe_layer = (
            config.moe_layer_freq > 0 and
            (layer_idx + 1) % config.moe_layer_freq == 0 and
            layer_idx > config.first_k_dense_replace
        )
        
        if is_moe_layer:
            self.mlp = DeepSeekV2MoE(config)
        else:
            self.mlp = DeepSeekV2MLP(config)
            
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        
        # Self Attention Block
        residual = hidden_states
        normalized_hidden_states = self.input_layernorm(hidden_states)
        
        attn_output, present_key_value = self.self_attn(
            hidden_states=normalized_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        
        hidden_states = residual + attn_output

        # MLP/MoE Block
        residual = hidden_states
        normalized_hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normalized_hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states, present_key_value

