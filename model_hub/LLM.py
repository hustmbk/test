import time
import torch
import torch.nn.functional as F
from termcolor import colored


class LLM:
    """
    A class representing the LLM (currently support Llama and Qwen).
    """

    def __init__(
        self, 
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str
    ) -> None:
        """ Initializes the LLM.
        Args:
            model_name (str): The name of the model.
            max_length (int): The maximum length (prefill+decode) of sequences.
            dtype (torch.dtype): The data type for model computations.
            device_map (str): The device for model, suppor 'cuda:x' or 'auto (automatically use all visible GPUs)'.
        """

        self.model_name = model_name
        self.max_length = max_length
        self.dtype = dtype
        self.device_map = device_map


    def layer_prefill(self, layer_idx, start_bdx, hidden_states):
        # print(f'Layer = {layer_idx}, start_bdx = {start_bdx}')

        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]
        
        # original hidden_states used as residual, clone a new one to process
        temp_hidden_states = hidden_states.clone()

        # chunk for lower memory comsumption
        for start_idx in range(0, seq_len, 8192//bsz):
            end_idx = min(seq_len, start_idx + 8192//bsz)
            temp_hidden_states[:, start_idx:end_idx, :] = self.layernorm(temp_hidden_states[:, start_idx:end_idx, :], 
                                                                         layer.input_layernorm_variance_epsilon, 
                                                                         layer.input_layernorm_weight)
        
        query_states, key_states, value_states = self.wqkv(temp_hidden_states, layer)
        del temp_hidden_states
        torch.cuda.empty_cache()
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)       # reshape [bs, seq_len, dim] => [bs, seq_len, head, head_dim]
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        key_states, value_states = self.kv_cache.prefill_update_kv_cache(query_states, key_states, value_states, layer_idx, start_bdx)
        torch.cuda.empty_cache()

        temp_attn_out = self.prefill_attention(query_states, key_states, value_states)

        self.kv_cache.sync(layer_idx, start_bdx)

        del query_states, key_states, value_states
        torch.cuda.empty_cache()

        hidden_states += self.wo(temp_attn_out, layer, temp_attn_out.shape[0], seq_len, dim)
        del temp_attn_out
        torch.cuda.empty_cache()

        # post attention
        residual = hidden_states.clone()

        # chunk for lower memory comsumption
        for start_idx in range(0, seq_len, 8192//bsz):
            end_idx = min(seq_len, start_idx + 8192//bsz)
            hidden_states[:, start_idx:end_idx, :] = self.layernorm(hidden_states[:, start_idx:end_idx, :], 
                                                                    layer.post_attention_layernorm_variance_epsilon, 
                                                                    layer.post_attention_layernorm_weight)
            hidden_states[:, start_idx:end_idx, :] = self.mlp(hidden_states[:, start_idx:end_idx, :], layer)   
        
        hidden_states += residual

        del residual
        torch.cuda.empty_cache()
                                                                                                   
        return hidden_states


    def layer_decode(self, layer_idx, hidden_states):
        # print(f'Layer = {layer_idx}')

        residual = hidden_states
        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]

        hidden_states = self.layernorm(hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        
        query_states, key_states, value_states = self.wqkv(hidden_states, layer)
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)

        key_states, value_states = self.kv_cache.decode_update_kv_cache(key_states, value_states, layer_idx)
        attn_out = self.decode_attention(query_states, key_states, value_states, layer_idx)
        hidden_states = self.wo(attn_out, layer, bsz, seq_len, dim)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layernorm(hidden_states, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        hidden_states = self.mlp(hidden_states, layer)
        hidden_states = residual + hidden_states

        return hidden_states


    def prefill_forward(self, inputs_ids):
        bsz, seq_len = inputs_ids.shape
        device = inputs_ids.device

        last_hidden_states = torch.empty((bsz, 1, self.hidden_size), dtype=self.dtype, device=device)
        for start_bdx in range(0, bsz, 1):
            end_bdx = min(bsz, start_bdx + 1)
            hidden_states = self.word_embedding(inputs_ids[start_bdx:end_bdx])  # [1, seq_len, hidden_size]

            if self.num_gpus > 1:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    hidden_states = self.parameter_move(hidden_states, ldx)
                    torch.cuda.empty_cache()
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :].to(self.layers[0].device)
            else:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    torch.cuda.empty_cache()
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :]
        
        last_hidden_states = self.layernorm(last_hidden_states.contiguous(), self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(last_hidden_states)
        
        return logits
        

    def decode_forward(self, inputs_ids):
        hidden_states = self.word_embedding(inputs_ids)

        if self.num_gpus > 1:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
                hidden_states = self.parameter_move(hidden_states, ldx)
            hidden_states = hidden_states.to(self.layers[0].device)
        else:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
        
        hidden_states = self.layernorm(hidden_states[:, -1:, :], self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(hidden_states)
        
        return logits


    def inference(self, inputs_ids):
        outputs_ids = []    # multi iteration, multi request
        output_ids = []     # single iteration, multi request
        
        # Get EOS token ID with fallback options
        eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
        if eos_token_id is None:
            # Try common EOS tokens
            try:
                eos_token_id = self.tokenizer.convert_tokens_to_ids('</s>')
            except:
                try:
                    eos_token_id = self.tokenizer.convert_tokens_to_ids('<|endoftext|>')
                except:
                    eos_token_id = None
        
        print(f"Using EOS token ID: {eos_token_id}")
        
        print("Start prefilling ...")
        torch.cuda.synchronize()
        prefill_start = time.time()

        logits = self.prefill_forward(inputs_ids=inputs_ids)
        output_ids = logits.argmax(dim=-1)
        outputs_ids.append(output_ids)
        self.move()

        torch.cuda.synchronize()
        prefill_end = time.time()
        print(colored(f"Prefilling latency: {round((prefill_end - prefill_start), 4)} s\n", 'green'))

        print("Start decoding ...")
        decode_start = time.time()

        # Track generated tokens for repetition penalty and pattern detection
        all_generated_ids = []
        
        for step in range(self.max_new_length-1):
            logits = self.decode_forward(inputs_ids=output_ids)
            
            # Apply enhanced repetition penalty with stronger settings
            logits = self.apply_repetition_penalty(logits, all_generated_ids, penalty=1.8, window_size=30)
            
            # Use improved sampling with more conservative settings
            output_ids = self.sample_next_token(logits, temperature=0.8, top_p=0.9, do_sample=True)
            
            outputs_ids.append(output_ids)
            current_token_ids = output_ids.flatten().tolist()
            all_generated_ids.extend(current_token_ids)
            
            # EOS token check
            if eos_token_id is not None and eos_token_id in current_token_ids:
                print(f"EOS token detected at step {step}, stopping generation...")
                break
            
            # Enhanced repetition pattern detection with aggressive stopping
            if len(all_generated_ids) >= 6:
                # Check for immediate token repetition 
                last_3 = all_generated_ids[-3:]
                if len(set(last_3)) == 1:  # Same token repeated 3 times
                    print(f"Immediate repetition detected at step {step}, stopping generation...")
                    break
                
                # Check for 2-token cycles
                if len(all_generated_ids) >= 8:
                    last_8 = all_generated_ids[-8:]
                    if (last_8[0] == last_8[2] == last_8[4] == last_8[6] and
                        last_8[1] == last_8[3] == last_8[5] == last_8[7]):
                        print(f"2-token cycle detected at step {step}, stopping generation...")
                        break
                
                # Check for suspicious patterns (more than 30% repeated in last 10 tokens)
                if len(all_generated_ids) >= 10:
                    last_10 = all_generated_ids[-10:]
                    unique_ratio = len(set(last_10)) / len(last_10)
                    if unique_ratio < 0.4:  # Less than 40% unique tokens
                        print(f"Low diversity detected (ratio={unique_ratio:.2f}) at step {step}, stopping generation...")
                        break

        decode_end = time.time()
        print(colored(
            f"Decoding latency: {round((decode_end - decode_start) * 1000 / (self.max_new_length - 1), 2)} ms/step, "
            f"Throughput: {round(self.batch_size * (self.max_new_length - 1) / (decode_end - decode_start), 2)} tokens/s\n",
            'green'
        ))
        
        outputs_ids = torch.cat(outputs_ids, dim=-1).tolist()
        
        return outputs_ids


    def apply_repetition_penalty(self, logits, generated_ids, penalty=1.2, window_size=50):
        """Apply enhanced repetition penalty to logits with frequency and recency weighting"""
        if not generated_ids:
            return logits
        
        # Only consider recent tokens for penalty (sliding window)
        recent_ids = generated_ids[-window_size:] if len(generated_ids) > window_size else generated_ids
        
        # Count token frequencies and track positions
        token_info = {}
        for i, token_id in enumerate(recent_ids):
            if token_id not in token_info:
                token_info[token_id] = {'count': 0, 'positions': []}
            token_info[token_id]['count'] += 1
            token_info[token_id]['positions'].append(i)
        
        # Apply penalty based on frequency, recency, and clustering
        vocab_size = logits.size(-1)
        for token_id, info in token_info.items():
            count = info['count']
            positions = info['positions']
            
            if count > 1 and token_id < vocab_size:  # Apply penalty for repeated tokens
                # Base penalty scales with frequency
                frequency_penalty = penalty + (count - 1) * 0.15
                
                # Recency penalty: more recent repetitions get higher penalty
                if positions:
                    recency_factor = sum([(pos + 1) / len(recent_ids) for pos in positions[-3:]]) / min(3, len(positions))
                    recency_penalty = 1.0 + recency_factor * 0.3
                else:
                    recency_penalty = 1.0
                
                # Clustering penalty: consecutive repetitions get extra penalty
                clustering_penalty = 1.0
                if len(positions) >= 2:
                    consecutive_count = 0
                    for j in range(1, len(positions)):
                        if positions[j] - positions[j-1] == 1:
                            consecutive_count += 1
                    if consecutive_count > 0:
                        clustering_penalty = 1.0 + consecutive_count * 0.2
                
                # Combined penalty 
                total_penalty = frequency_penalty * recency_penalty * clustering_penalty
                
                # Apply penalty to logits
                if logits[0, 0, token_id] > 0:
                    logits[0, 0, token_id] /= total_penalty
                else:
                    logits[0, 0, token_id] *= total_penalty
        
        return logits

    def sample_next_token(self, logits, temperature=1.0, top_p=0.9, do_sample=True):
        """Sample next token with improved temperature and nucleus sampling"""
        if not do_sample:
            # Greedy decoding
            return logits.argmax(dim=-1)
        
        # Apply temperature scaling (with safety checks)
        if temperature > 0 and temperature != 1.0:
            logits = logits / max(temperature, 1e-7)  # Prevent division by zero
        
        # Add small random noise to break ties and reduce determinism
        if temperature > 0:
            noise = torch.randn_like(logits) * 0.01
            logits = logits + noise
        
        # Apply top-p (nucleus) sampling with improvements
        if top_p < 1.0 and top_p > 0:
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            
            # Calculate cumulative probabilities
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Find tokens to remove (above threshold)
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # Keep at least the top token
            sorted_indices_to_remove[..., 0] = False
            
            # Shift to include the first token above threshold
            if sorted_indices_to_remove.size(-1) > 1:
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
            
            # Create mask for original indices
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            
            # Set removed tokens to very low probability
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Final safety check: ensure we don't have all -inf logits
        if torch.all(torch.isinf(logits)):
            print("Warning: All logits are -inf, falling back to greedy decoding")
            return torch.zeros_like(logits[..., 0]).long()
        
        # Sample from the filtered distribution
        try:
            probs = F.softmax(logits, dim=-1)
            # Add small epsilon to prevent zero probabilities
            probs = probs + 1e-8
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            next_token = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
            return next_token.view(logits.shape[:-1])
        except RuntimeError as e:
            print(f"Sampling error: {e}, falling back to greedy decoding")
            return logits.argmax(dim=-1)


    def generate(self, attention_type, inputs_ids, attention_masks, max_new_length, attn_config=None):
        """ LLM Inference.
        Args:
            attention_type: str,
            input_ids (torch.tensor): The input of LLM.
            attention_masks (torch.tensor): The attention masks of LLM.
            max_new_length (int): The maximum length of generated sequences.
        """

        bs, input_length = inputs_ids.shape
        assert input_length + max_new_length <= self.max_length, \
        f"Error: input_length({input_length}) + max_new_length({max_new_length}) exceeds max_length({self.max_length})"

        self.batch_size = bs
        self.input_length = input_length
        self.max_new_length = max_new_length
        self.attention_type = attention_type

        valid_start = attention_masks.shape[1] - torch.sum(attention_masks, dim=-1).detach().cpu().numpy()
        del attention_masks
        torch.cuda.empty_cache()

        print("Allocate GPU buffers and CPU pin memory ...\n")
        self.init_kv_cache(input_length, valid_start, attn_config)

        outputs = self.inference(inputs_ids)

        return outputs
