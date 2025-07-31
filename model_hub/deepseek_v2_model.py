import torch
import torch.nn as nn
from pathlib import Path
from safetensors import safe_open
from termcolor import colored
import time

from LLM import LLM # Assuming LLM base class is in the parent directory
from .deepseek_v2_config import DeepSeekV2Config
from .deepseek_v2_layer import DeepSeekV2DecoderLayer
from .deepseek_v2_attention import RMSNorm

class DeepSeekV2Model(LLM, nn.Module):
    """
    A self-contained implementation of the DeepSeek-V2 model.
    This class handles model architecture, weight loading, and generation
    without relying on transformers.AutoModelForCausalLM.
    """
    def __init__(self, model_path: str, max_length: int, dtype: torch.dtype, device_map: str):
        # Note: device_map is simplified to use a single device for this implementation
        super(LLM, self).__init__(model_name=model_path, max_length=max_length, dtype=dtype, device_map=device_map)
        nn.Module.__init__(self)

        self.device = torch.device(device_map if "cuda" in device_map else "cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        print("Initializing self-contained DeepSeekV2Model...")
        self.config = DeepSeekV2Config.from_pretrained(model_path)
        
        # Build model architecture
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size, self.config.pad_token_id)
        self.layers = nn.ModuleList([DeepSeekV2DecoderLayer(self.config, i) for i in range(self.config.num_hidden_layers)])
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # Move model to device
        self.to(self.device, dtype=self.dtype)
        
        # Load weights
        self.load_weights(model_path)
        
    def load_weights(self, model_path: str):
        """Loads weights from .safetensors files into the model."""
        print(f"Loading weights from {model_path}...")
        path = Path(model_path)
        files = list(path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")

        state_dict = {}
        for f in files:
            with safe_open(f, framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    state_dict[key] = sf.get_tensor(key)
        
        # The load_state_dict method will correctly map the keys
        self.load_state_dict(state_dict, strict=True)
        print(colored("✅ All weights loaded successfully.", 'green'))
        
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        
        hidden_states = self.embed_tokens(input_ids)
        
        next_kv_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            
            hidden_states, present_kv = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            
            if use_cache:
                next_kv_cache.append(present_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits, next_kv_cache

    def generate(self, inputs_ids, attention_masks, max_new_length, **kwargs):
        """
        Generates sequences of token ids for models with a language modeling head.
        """
        batch_size, prompt_len = inputs_ids.shape
        self.eval() # Set model to evaluation mode

        # Prefill phase
        print("Prefilling...")
        torch.cuda.synchronize()
        prefill_start_time = time.time()
        
        position_ids = torch.arange(0, prompt_len, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, past_key_values = self.forward(
                input_ids=inputs_ids,
                position_ids=position_ids,
                attention_mask=attention_masks,
                use_cache=True,
            )
        
        torch.cuda.synchronize()
        prefill_end_time = time.time()
        print(colored(f"Prefill latency: {round((prefill_end_time - prefill_start_time) * 1000, 2)} ms", 'green'))

        # Decode phase
        print("Decoding...")
        generated_ids = []
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        
        decode_start_time = time.time()
        for _ in range(max_new_length):
            generated_ids.append(next_token)
            
            current_len = next_token.shape[1] + past_key_values[0][0].shape[2]
            position_ids = torch.arange(current_len - 1, current_len, dtype=torch.long, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                logits, past_key_values = self.forward(
                    input_ids=next_token,
                    position_ids=position_ids,
                    attention_mask=None, # In decode, mask is handled by causal attention
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            
            next_token = torch.argmax(logits, dim=-1)
            
            if self.config.eos_token_id is not None and next_token.item() == self.config.eos_token_id:
                print("EOS token detected. Stopping generation.")
                break
        
        torch.cuda.synchronize()
        decode_end_time = time.time()
        num_decoded = len(generated_ids)
        if num_decoded > 0:
             print(colored(f"Decoding latency: {round((decode_end_time - decode_start_time) * 1000 / num_decoded, 2)} ms/token", 'green'))

        return torch.cat(generated_ids, dim=1) if generated_ids else torch.tensor([[]], dtype=torch.long)

