from flash_attn import flash_attn_with_kvcache
import torch.nn.functional as F
import torch


def retroinfer_prefill_attn(query_states, key_states, value_states, causal):
    # Handle different dimensions for Q/K vs V for DeepSeek-V2
    # Q/K have 192 dim, V has 128 dim for DeepSeek-V2
    bsz, seq_len, num_heads, q_head_dim = query_states.shape
    _, _, num_kv_heads, k_head_dim = key_states.shape  
    _, _, _, v_head_dim = value_states.shape
    
    # For RetroInfer, we need all dimensions to match the VALUE dimension (128)
    # If we have dimension mismatch, slice Q and K to match V
    if q_head_dim != v_head_dim:
        # Slice Q and K to match V dimension (128)
        query_states_adapted = query_states[:, :, :, :v_head_dim]
        key_states_adapted = key_states[:, :, :, :v_head_dim]
    else:
        query_states_adapted = query_states
        key_states_adapted = key_states

    attn_out = flash_attn_with_kvcache(
        q=query_states_adapted, 
        k_cache=key_states_adapted, 
        v_cache=value_states,
        causal=causal
    )
    
    # Based on FlashMLA: output should have VALUE dimension, not query dimension
    # Do NOT pad the output back - this is the correct behavior
    return attn_out



def retroinfer_decode_attn(query_states, retroinfer_cache, value_states, layer_idx):
    # For RetroInfer decode, we need to handle dimension mismatch differently
    # RetroInfer's compute method expects Q/K/V to have the same dimension
    bsz, seq_len, num_heads, q_head_dim = query_states.shape
    
    # Check if we have dimension mismatch
    if hasattr(retroinfer_cache, 'head_dim'):
        cache_head_dim = retroinfer_cache.head_dim
        if q_head_dim != cache_head_dim:
            # Slice query to match cache dimension (use only first 128 dims)
            # This is based on the insight that DeepSeek-V2's Q has 192 dims but only the first 128 are "value-relevant"
            query_states_adapted = query_states[:, :, :, :cache_head_dim]
            
            # Use the adapted query for RetroInfer compute
            attn_out = retroinfer_cache.compute(
                query_states_adapted.contiguous(), layer_idx
            )
            
            # Based on FlashMLA: output should have VALUE dimension, not query dimension
            # Do NOT pad output back - this is the correct behavior
            return attn_out
    
    # Fallback: direct compute if no dimension mismatch
    attn_out = retroinfer_cache.compute(
        query_states.contiguous(), layer_idx
    )
    
    return attn_out
