# llama.py - Llama系列模型的具体实现
# 继承自LLM基类，实现了Llama模型特定的方法
# 支持多GPU分布式推理和两种注意力机制（Flash Attention和RetroInfer）

import gc
import re
import os
import json
import torch
import torch.nn.functional as F
import flashinfer
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
from .LLM import LLM
from cache_hub import flash_attn_cache, retroinfer_cache
from attn_hub import prefill_full_flash_attn, decode_full_flash_attn, retroinfer_prefill_attn, retroinfer_decode_attn



class LlamaLayer:
    """
    Llama模型的单层封装
    
    功能：
    - 管理单层的所有参数（注意力权重、FFN权重、LayerNorm权重）
    - 支持权重融合优化（如将Q、K、V权重合并为QKV权重）
    - 支持设备分配（可以将不同层放在不同GPU上）
    """

    def __init__(self, layer_idx, device) -> None:
        """
        初始化Llama层
        
        参数:
            layer_idx: 层索引
            device: 该层所在的设备（如'cuda:0'）
        """
        self.layer_idx = layer_idx
        self.device = device
    
    def init_layer(self, hf_llama_layer):
        """
        从HuggingFace的Llama层初始化权重
        
        参数:
            hf_llama_layer: HuggingFace格式的Llama层
            
        优化策略:
        1. 将Q、K、V三个投影矩阵合并为一个QKV矩阵，减少矩阵乘法次数
        2. 将Gate和Up投影矩阵合并，优化FFN计算
        3. 将权重移动到指定设备并使用non_blocking实现异步传输
        4. 删除原始分离的权重以节省内存
        """
        # 提取注意力权重
        self.wq = hf_llama_layer.self_attn.q_proj.weight.detach()
        self.wk = hf_llama_layer.self_attn.k_proj.weight.detach()
        self.wv = hf_llama_layer.self_attn.v_proj.weight.detach()
        # 合并QKV权重以优化计算
        self.wqkv = torch.cat((self.wq, self.wk, self.wv), dim=0).to(self.device, non_blocking=True)
        self.wo = hf_llama_layer.self_attn.o_proj.weight.detach().to(self.device, non_blocking=True)

        # 提取FFN权重
        self.gate_proj = hf_llama_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_llama_layer.mlp.up_proj.weight.detach()
        # 合并Gate和Up权重以优化计算
        self.gate_up_proj = torch.cat((self.gate_proj, self.up_proj), dim=0).to(self.device, non_blocking=True)
        self.down_proj = hf_llama_layer.mlp.down_proj.weight.detach().to(self.device, non_blocking=True)

        # 提取LayerNorm权重
        self.input_layernorm_weight = hf_llama_layer.input_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.input_layernorm_variance_epsilon = hf_llama_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_llama_layer.post_attention_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.post_attention_layernorm_variance_epsilon = hf_llama_layer.post_attention_layernorm.variance_epsilon

        # 删除原始分离的权重以节省内存
        del self.wq, self.wk, self.wv, self.gate_proj, self.up_proj


class LlamaModel(LLM):
    """
    Llama模型的具体实现
    
    特点：
    1. 支持Llama系列所有模型（Llama-2, Llama-3等）
    2. 实现了RMSNorm归一化
    3. 使用RoPE（旋转位置编码）
    4. 支持GQA（分组查询注意力）优化
    5. 集成FlashInfer库进行高效计算
    """

    def __init__(
        self, 
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str
    ) -> None:
        """
        初始化Llama模型
        
        参数:
            model_name: 模型名称或路径
            max_length: 最大序列长度
            dtype: 数据类型
            device_map: 设备映射策略
        """
        super().__init__(model_name, max_length, dtype, device_map)

        # 加载tokenizer和配置
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = LlamaConfig.from_pretrained(model_name)
        
        # 提取模型架构参数
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        # GQA优化：计算每个KV头对应的查询头数量
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = self.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = self.config.max_position_embeddings
        self.vocab_size = self.config.vocab_size
        self.eos_tokens = [self.config.eos_token_id]

        # 初始化模型权重
        self.init_model()


    def _set_cos_sin_cache(self):
        """
        预计算RoPE的cos和sin缓存
        
        返回:
            cos_cache, sin_cache: 预计算的位置编码缓存
            
        注意：
        - 预计算可以避免推理时的重复计算
        - 缓存大小基于max_length而非max_position_embeddings
        - 应用了attention_scaling以支持长序列
        """
        t = torch.arange(self.max_length, device=self.inv_freq.device, dtype=self.dtype)  # 使用模型数据类型
        freqs = torch.outer(t, self.inv_freq.to(self.dtype))  # 确保inv_freq也是正确类型
        return freqs.cos()*self.attention_scaling, freqs.sin()*self.attention_scaling


    def init_model(self):
        """
        初始化模型权重和参数
        
        处理流程：
        1. 从HuggingFace加载预训练模型
        2. 根据device_map决定单GPU还是多GPU部署
        3. 将模型权重分配到相应设备
        4. 初始化位置编码缓存
        5. 逐层初始化并释放原始模型以节省内存
        """
        # 加载预训练模型
        hf_llama = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype=self.dtype)

        # 确定GPU数量和设备映射策略
        self.num_gpus = torch.cuda.device_count() if self.device_map == 'auto' else 1
        if self.device_map == 'auto' and self.num_gpus == 1:
            self.device_map = 'cuda:0'
        
        if self.device_map != "auto":   # 单GPU部署
            # 创建层到设备的映射（所有层都在同一设备）
            self.layer_mapping = {}
            for ldx in range(0, self.num_layers):
                self.layer_mapping.update({str(ldx): self.device_map})

            # 将embedding和lm_head移到目标设备
            self.embed_tokens = hf_llama.model.embed_tokens.weight.detach().to(self.device_map, non_blocking=True)
            self.lm_head = hf_llama.lm_head.weight.detach().to(self.device_map, non_blocking=True)

            # 初始化最终的LayerNorm
            self.norm_weight = hf_llama.model.norm.weight.detach().to(self.device_map, non_blocking=True)
            self.norm_variance_epsilon = hf_llama.model.norm.variance_epsilon

            # 初始化位置编码相关参数
            self.position_ids = torch.arange(0, self.max_length).to(self.device_map, non_blocking=True)
            self.inv_freq = hf_llama.model.rotary_emb.inv_freq.detach().to(self.device_map, non_blocking=True)
            self.attention_scaling = hf_llama.model.rotary_emb.attention_scaling
            # 预计算位置编码缓存
            self.cos_cache, self.sin_cache = self._set_cos_sin_cache()
            self.cos_sin_cache = torch.cat((self.cos_cache, self.sin_cache), dim=-1)

            # 逐层初始化
            self.layers = []
            for idx, hf_llama_layer in enumerate(hf_llama.model.layers):
                llama_layer = LlamaLayer(idx, device=self.device_map)
                llama_layer.init_layer(hf_llama_layer)
                self.layers.append(llama_layer)
                # 立即释放原始层以节省内存
                hf_llama.model.layers[idx] = None

        else:                         # 多GPU部署
            # 计算层分配策略
            self.gpu_ids = list(range(self.num_gpus))
            self.layer_interval = (self.num_layers + self.num_gpus - 1) // self.num_gpus
            self.layer_mapping = {}
            # 均匀分配层到各个GPU
            for ldx in range(0, self.num_layers):
                self.layer_mapping.update({str(ldx): f'cuda:{ldx // self.layer_interval}'})

            # embedding和lm_head放在第一个GPU
            self.embed_tokens = hf_llama.model.embed_tokens.weight.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.lm_head = hf_llama.lm_head.weight.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)

            self.norm_weight = hf_llama.model.norm.weight.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.norm_variance_epsilon = hf_llama.model.norm.variance_epsilon

            # 位置编码参数也放在第一个GPU
            self.position_ids = torch.arange(0, self.max_length).to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.inv_freq = hf_llama.model.rotary_emb.inv_freq.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.attention_scaling = hf_llama.model.rotary_emb.attention_scaling
            self.cos_cache, self.sin_cache = self._set_cos_sin_cache()
            self.cos_sin_cache = torch.cat((self.cos_cache, self.sin_cache), dim=-1)

            # 根据层映射初始化各层
            self.layers = []
            for ldx, hf_llama_layer in enumerate(hf_llama.model.layers):
                llama_layer = LlamaLayer(ldx, device=self.layer_mapping[str(ldx)])
                llama_layer.init_layer(hf_llama_layer)
                self.layers.append(llama_layer)
                hf_llama.model.layers[ldx] = None

        # 清理临时变量并释放内存
        del self.inv_freq, self.cos_cache, self.sin_cache
        gc.collect()
        torch.cuda.empty_cache()


    def init_kv_cache(self, real_input_length, valid_start, attn_config=None):
        """
        初始化KV缓存
        
        参数:
            real_input_length: 实际输入长度
            valid_start: 有效序列的起始位置
            attn_config: 注意力配置（如果为None，从配置文件读取）
            
        支持两种缓存类型：
        1. Flash Attention缓存：全部存储在GPU上
        2. RetroInfer缓存：GPU-CPU混合存储，支持更长序列
        """
        if attn_config is None:
            # 从配置文件读取默认配置
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
            PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
            CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
            MODEL_NAME = self.model_name.split("/")[-1]+'.json'
            CONFIG_FILE = os.path.join(CONFIG_DIR, MODEL_NAME)

            with open(CONFIG_FILE, "r") as f:
                llama_config = json.load(f)
        else:
            llama_config = attn_config
        
        # 根据注意力类型初始化相应的缓存
        if self.attention_type == 'Full_Flash_Attn':
            # Flash Attention：全GPU缓存
            self.kv_cache = flash_attn_cache(
                valid_start = valid_start,
                layer_num = self.num_layers,
                batch_size = self.batch_size,
                max_length = self.max_new_length + real_input_length,
                num_key_value_heads = self.num_key_value_heads,
                num_heads = self.num_heads,
                head_dim = self.head_dim,
                dtype = self.dtype,
                layer_mapping = self.layer_mapping,
                num_gpus = self.num_gpus,
                model_size = int(re.search(r'(\d+)[B]', self.model_name).group(1))
            )
        elif self.attention_type == 'RetroInfer':
            # RetroInfer：GPU-CPU混合缓存，支持更长序列
            retroinfer_config = llama_config.get(self.attention_type)

            self.kv_cache = retroinfer_cache(
                valid_start = valid_start,
                layer_num = self.num_layers,
                batch_size = self.batch_size,
                max_length = self.max_new_length + real_input_length,
                num_key_value_heads = self.num_key_value_heads,
                num_heads = self.num_heads,
                head_dim = self.head_dim,
                dtype = self.dtype,
                layer_mapping = self.layer_mapping,
                max_new_length = self.max_new_length,
                # RetroInfer特定参数
                static_pattern_start = retroinfer_config["static_pattern_start"],
                static_pattern_end = retroinfer_config["static_pattern_end"],
                core = retroinfer_config["core"],
                n_centroids = retroinfer_config["n_centroids"],
                n_segment = retroinfer_config["n_segment"],
                nprobe = retroinfer_config["nprobe"],
                max_compute_cluster_num = retroinfer_config["max_compute_cluster_num"],
                cache_unit_size = retroinfer_config["cache_unit_size"],
                cache_cluster_num = retroinfer_config["cache_cluster_num"],
                num_gpus = self.num_gpus,
                model_size = int(re.search(r'(\d+)[B]', self.model_name).group(1))
            )
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")

    
    def move(self):
        """
        执行必要的数据移动操作
        
        功能：
        - Flash Attention：将KV缓存移到GPU
        - RetroInfer：准备CPU-GPU混合缓存
        """
        torch.cuda.empty_cache()
        if self.attention_type == 'Full_Flash_Attn':
            self.kv_cache.move_gpu()
        elif self.attention_type == 'RetroInfer':
            self.kv_cache.prepare_cache()
        torch.cuda.empty_cache()

    
    def word_embedding(self, inputs_id):
        """
        词嵌入层
        
        参数:
            inputs_id: 输入的token ID
            
        返回:
            嵌入向量
        """
        hidden_states = F.embedding(inputs_id, self.embed_tokens)
        return hidden_states

    
    def lm(self, hidden_states):
        """
        语言模型头（将隐藏状态映射到词表）
        
        参数:
            hidden_states: 模型的隐藏状态
            
        返回:
            logits: 词表上的概率分布
        """
        logits = F.linear(hidden_states, self.lm_head).float()
        return logits


    def wqkv(self, hidden_states, layer):
        """
        计算Query、Key、Value矩阵
        
        参数:
            hidden_states: 输入的隐藏状态
            layer: 当前层
            
        返回:
            query_states, key_states, value_states
            
        优化：
        - 使用合并的QKV权重矩阵，只需一次矩阵乘法
        - 支持GQA优化（KV头数少于Q头数）
        """
        # 一次矩阵乘法计算QKV
        qkv = F.linear(hidden_states, layer.wqkv)
        # 根据GQA配置分割QKV
        query_states, key_states, value_states = qkv.split(
            [self.hidden_size, 
             self.hidden_size//self.num_key_value_groups, 
             self.hidden_size//self.num_key_value_groups], 
            dim=-1
        )
        return query_states, key_states, value_states

    
    def wo(self, hidden_states, layer, bsz, seq_len, dim):
        """
        输出投影层
        
        参数:
            hidden_states: 注意力输出
            layer: 当前层
            bsz, seq_len, dim: 张量维度
            
        返回:
            投影后的隐藏状态
        """
        hidden_states = hidden_states.reshape(bsz, seq_len, dim)
        hidden_states = F.linear(hidden_states, layer.wo)
        return hidden_states

    
    def prefill_attention(self, query_states, key_states, value_states):
        """
        预填充阶段的注意力计算
        
        参数:
            query_states, key_states, value_states: QKV矩阵
            
        返回:
            注意力输出
            
        注意：
        - 使用因果掩码（causal=True）
        - 根据attention_type选择不同的实现
        """
        if self.attention_type == 'Full_Flash_Attn':
            attn_out = prefill_full_flash_attn(query_states, key_states, value_states, causal=True)
        elif self.attention_type == 'RetroInfer':
            attn_out = retroinfer_prefill_attn(query_states, key_states, value_states, causal=True)
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        return attn_out
    

    def decode_attention(self, query_states, key_states, value_states, layer_idx):
        """
        解码阶段的注意力计算
        
        参数:
            query_states, key_states, value_states: QKV矩阵
            layer_idx: 层索引
            
        返回:
            注意力输出
            
        注意：
        - 使用已有的KV缓存
        - 只需计算新token与历史序列的注意力
        """
        if self.attention_type == 'Full_Flash_Attn':
            attn_out = decode_full_flash_attn(query_states, key_states, value_states, layer_idx, self.kv_cache)
        elif self.attention_type == 'RetroInfer':
            attn_out = retroinfer_decode_attn(query_states, key_states, value_states, layer_idx, self.kv_cache)
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        return attn_out

    
    def mlp(self, hidden_states, layer):
        """
        前馈网络（FFN）层
        
        参数:
            hidden_states: 输入的隐藏状态
            layer: 当前层
            
        返回:
            FFN输出
            
        实现细节：
        - 使用SiLU激活函数
        - Gate和Up投影已合并，通过FlashInfer的silu_and_mul实现高效计算
        """
        # 计算gate和up投影
        hidden_states = F.linear(hidden_states, layer.gate_up_proj)
        dim = hidden_states.shape[-1] // 2
        hidden_shape = (hidden_states.shape[:-1] + (dim,))
        out = torch.empty(hidden_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        # 使用FlashInfer的融合算子：SiLU(gate) * up
        flashinfer.activation.silu_and_mul(hidden_states, out)
        # down投影
        hidden_states = F.linear(out, layer.down_proj)
        return hidden_states 

    
    def parameter_move(self, hidden_states, ldx):
        """
        多GPU场景下的参数移动
        
        参数:
            hidden_states: 当前的隐藏状态
            ldx: 当前层索引
            
        返回:
            移动到下一设备的隐藏状态
            
        功能：
        1. 确定下一层所在的设备
        2. 移动hidden_states到该设备
        3. 移动必要的辅助张量（位置编码、缓存索引等）
        """
        # 确定下一层的设备
        next_device = self.layer_mapping[str(ldx+1)] if str(ldx+1) in self.layer_mapping else self.layer_mapping[str(0)]
        torch.cuda.set_device(next_device)
        
        # 移动主要张量
        hidden_states = hidden_states.to(next_device)
        self.position_ids = self.position_ids.to(next_device)
        self.cos_sin_cache = self.cos_sin_cache.to(next_device)
        
        # 根据注意力类型移动相应的辅助张量
        if self.attention_type == 'Full_Flash_Attn':
            if hidden_states.shape[1] == 1:  # 解码阶段
                self.kv_cache.batch_indices = self.kv_cache.batch_indices.to(next_device)
                self.kv_cache.valid_length = self.kv_cache.valid_length.to(next_device)
        elif self.attention_type == 'RetroInfer':
            if hidden_states.shape[1] == 1:  # 解码阶段
                # 移动RetroInfer的各种缓冲区
                self.kv_cache.gemm_o = self.kv_cache.gemm_o.to(next_device)
                self.kv_cache.softmax_o = self.kv_cache.softmax_o.to(next_device)
                self.kv_cache.norm = self.kv_cache.norm.to(next_device)
                self.kv_cache.sum = self.kv_cache.sum.to(next_device)
                self.kv_cache.es_centroids = self.kv_cache.es_centroids.to(next_device)
                self.kv_cache.es_value_sum = self.kv_cache.es_value_sum.to(next_device)
                self.kv_cache.es_cluster_size = self.kv_cache.es_cluster_size.to(next_device)
                self.kv_cache.execution_buffer_keys = self.kv_cache.execution_buffer_keys.to(next_device)
                self.kv_cache.execution_buffer_values = self.kv_cache.execution_buffer_values.to(next_device)
                self.kv_cache.valid_lengths = self.kv_cache.valid_lengths.to(next_device)
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        return hidden_states

    
    def layernorm(self, hidden_states, epsilon, weight):
        """
        RMSNorm归一化
        
        参数:
            hidden_states: 输入张量
            epsilon: 防止除零的小值
            weight: 归一化权重
            
        返回:
            归一化后的张量
            
        注意：
        - Llama使用RMSNorm而非LayerNorm
        - 使用FlashInfer的高效实现
        """
        bsz, seq_len, dim = hidden_states.shape
        hidden_states = hidden_states.reshape(bsz * seq_len, dim)
        hidden_states = flashinfer.rmsnorm(hidden_states, weight, epsilon)
        hidden_states = hidden_states.reshape(bsz, seq_len, dim)
        return hidden_states


    def apply_rotary_pos_emb(self, query_states, key_states, position_ids):
        """
        应用旋转位置编码（RoPE）
        
        参数:
            query_states: 查询向量
            key_states: 键向量
            position_ids: 位置索引
            
        返回:
            应用RoPE后的查询和键向量
            
        优化：
        - 使用预计算的cos/sin缓存
        - 使用FlashInfer的in-place实现减少内存分配
        """
        bsz, _, hidden_dim = query_states.shape
        _, _, kv_dim = key_states.shape
        
        # 展平以应用RoPE
        query_states = query_states.view(-1, hidden_dim)
        key_states = key_states.view(-1, kv_dim)
        
        # FlashInfer RoPE API需要一维position_ids，将[bsz, seq_len]转换为一维
        position_ids_flat = position_ids.reshape(-1)
        
        # 使用FlashInfer的高效RoPE实现
        flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(
            position_ids_flat, query_states, key_states, 
            self.head_dim, self.cos_sin_cache, True
        )
        
        # 恢复原始形状
        query_states = query_states.view(bsz, -1, hidden_dim)
        key_states = key_states.view(bsz, -1, kv_dim)
        return query_states, key_states


    def position_embedd(self, query_states, key_states):
        """
        位置编码的主接口
        
        参数:
            query_states: 查询向量
            key_states: 键向量
            
        返回:
            应用位置编码后的查询和键向量
            
        注意：
        - 根据当前上下文位置生成position_ids
        - 支持批处理（所有序列使用相同的位置）
        """
        bsz, seq_len, _ = key_states.shape

        # 生成位置ID（考虑当前的上下文位置）
        position_ids = self.position_ids[self.kv_cache.context:self.kv_cache.context+seq_len].unsqueeze(0).repeat(bsz, 1)
        
        # 应用RoPE
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)

        return query_states, key_states

    