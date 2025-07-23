# LLM.py - 大语言模型的基础类实现
# 提供了预填充(prefill)和解码(decode)的核心功能
# 支持多GPU并行计算和KV缓存管理

import time
import torch
from termcolor import colored
from utils.logger import get_logger, start_phase, end_phase, log_gpu_memory, log_error_with_context, log_inference_stats


class LLM:
    """
    大语言模型基础类，支持Llama和Qwen系列模型
    
    主要功能：
    1. 预填充(Prefill)：处理输入序列，生成初始的KV缓存
    2. 解码(Decode)：基于KV缓存逐步生成新的token
    3. 多GPU支持：自动处理模型在多GPU上的参数分布
    4. KV缓存管理：与cache_hub模块配合管理注意力机制的KV缓存
    
    注意：这是一个抽象基类，具体模型（如Llama、Qwen）需要继承并实现特定方法
    """

    def __init__(
        self, 
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str
    ) -> None:
        """ 
        初始化LLM模型
        
        参数:
            model_name: 模型名称（例如 'llama-3-8b'）
            max_length: 最大序列长度（预填充+解码的总长度）
            dtype: 模型计算的数据类型（如torch.float16）
            device_map: 设备映射，支持 'cuda:x' 或 'auto'（自动使用所有可见GPU）
        """
        logger = get_logger()
        logger.info("初始化LLM基础模型", 
                   model_name=model_name,
                   max_length=max_length,
                   dtype=str(dtype),
                   device_map=device_map)

        self.model_name = model_name
        self.max_length = max_length
        self.dtype = dtype
        self.device_map = device_map


    def layer_prefill(self, layer_idx, start_bdx, hidden_states):
        """
        预填充阶段的单层处理
        
        参数:
            layer_idx: 当前处理的层索引
            start_bdx: 批次的起始索引（用于批量处理时的定位）
            hidden_states: 输入的隐藏状态 [batch_size, seq_len, hidden_dim]
            
        返回:
            处理后的隐藏状态
            
        处理流程:
        1. LayerNorm归一化
        2. 计算QKV（查询、键、值）
        3. 位置编码
        4. 更新KV缓存
        5. 执行注意力计算
        6. 输出投影和残差连接
        7. FFN（前馈网络）处理
        
        注意：使用分块处理以降低内存消耗
        """
        logger = get_logger()
        logger.debug(f"开始处理第{layer_idx}层预填充", 
                    start_bdx=start_bdx, 
                    input_shape=hidden_states.shape)
        
        try:
            bsz, seq_len, dim = hidden_states.shape
            layer = self.layers[layer_idx]
            
            # 保留原始hidden_states作为残差，克隆一个新的用于处理
            temp_hidden_states = hidden_states.clone()

            # 分块处理以降低内存消耗
            # 每次处理8192个token（根据batch size调整）
            # 这种策略可以有效避免GPU内存溢出
            for start_idx in range(0, seq_len, 8192//bsz):
                end_idx = min(seq_len, start_idx + 8192//bsz)
                temp_hidden_states[:, start_idx:end_idx, :] = self.layernorm(temp_hidden_states[:, start_idx:end_idx, :], 
                                                                             layer.input_layernorm_variance_epsilon, 
                                                                             layer.input_layernorm_weight)
            
            # 计算查询(Q)、键(K)、值(V)矩阵
            query_states, key_states, value_states = self.wqkv(temp_hidden_states, layer)
            # 立即释放临时变量以节省内存
            del temp_hidden_states
            torch.cuda.empty_cache()
            # 应用旋转位置编码(RoPE)
            query_states, key_states = self.position_embedd(query_states, key_states)

            # 重塑张量形状以适应多头注意力计算
            query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)       # reshape [bs, seq_len, dim] => [bs, seq_len, head, head_dim]
            key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
            value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

            # 更新KV缓存（这是RetroInfer的核心优化部分）
            # 缓存会被存储在GPU或CPU上，取决于attention_type
            key_states, value_states = self.kv_cache.prefill_update_kv_cache(query_states, key_states, value_states, layer_idx, start_bdx)
            torch.cuda.empty_cache()

            # 执行注意力计算
            temp_attn_out = self.prefill_attention(query_states, key_states, value_states)

            # 同步KV缓存（对于CPU缓存，确保数据传输完成）
            self.kv_cache.sync(layer_idx, start_bdx)

            # 释放内存
            del query_states, key_states, value_states
            torch.cuda.empty_cache()

            # 输出投影并加上残差连接
            hidden_states += self.wo(temp_attn_out, layer, temp_attn_out.shape[0], seq_len, dim)
            del temp_attn_out
            torch.cuda.empty_cache()

            # Post-attention处理：LayerNorm + FFN + 残差连接
            residual = hidden_states.clone()

            # 分块处理FFN以降低内存消耗
            for start_idx in range(0, seq_len, 8192//bsz):
                end_idx = min(seq_len, start_idx + 8192//bsz)
                hidden_states[:, start_idx:end_idx, :] = self.layernorm(hidden_states[:, start_idx:end_idx, :], 
                                                                        layer.post_attention_layernorm_variance_epsilon, 
                                                                        layer.post_attention_layernorm_weight)
                hidden_states[:, start_idx:end_idx, :] = self.mlp(hidden_states[:, start_idx:end_idx, :], layer)   
            
            # 最终的残差连接
            hidden_states += residual

            del residual
            torch.cuda.empty_cache()
            
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
        解码阶段的单层处理
        
        参数:
            layer_idx: 当前处理的层索引
            hidden_states: 输入的隐藏状态 [batch_size, 1, hidden_dim]
            
        返回:
            处理后的隐藏状态
            
        与预填充的主要区别:
        1. 输入序列长度为1（每次只处理一个新token）
        2. 直接使用已有的KV缓存进行注意力计算
        3. 不需要分块处理（序列长度短）
        """
        # print(f'Layer = {layer_idx}')

        residual = hidden_states
        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]

        # Pre-attention处理
        hidden_states = self.layernorm(hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        
        # 计算QKV
        query_states, key_states, value_states = self.wqkv(hidden_states, layer)
        query_states, key_states = self.position_embedd(query_states, key_states)

        # 重塑张量形状
        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)

        # 更新KV缓存并执行注意力计算
        key_states, value_states = self.kv_cache.decode_update_kv_cache(key_states, value_states, layer_idx)
        attn_out = self.decode_attention(query_states, key_states, value_states, layer_idx)
        hidden_states = self.wo(attn_out, layer, bsz, seq_len, dim)
        hidden_states = residual + hidden_states

        # Post-attention处理
        residual = hidden_states
        hidden_states = self.layernorm(hidden_states, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        hidden_states = self.mlp(hidden_states, layer)
        hidden_states = residual + hidden_states

        return hidden_states


    def prefill_forward(self, inputs_ids):
        """
        预填充阶段的前向传播
        
        参数:
            inputs_ids: 输入的token ID [batch_size, seq_len]
            
        返回:
            最后一个位置的logits [batch_size, 1, vocab_size]
            
        处理流程:
        1. 词嵌入
        2. 逐层处理（支持多GPU）
        3. 最终的LayerNorm
        4. 语言模型头投影
        
        注意：为了节省内存，按批次逐个处理
        """
        bsz, seq_len = inputs_ids.shape
        device = inputs_ids.device

        # 存储每个批次的最后一个隐藏状态
        last_hidden_states = torch.empty((bsz, 1, self.hidden_size), dtype=self.dtype, device=device)
        # 逐批次处理以节省内存（每次处理1个序列）
        for start_bdx in range(0, bsz, 1):
            end_bdx = min(bsz, start_bdx + 1)
            hidden_states = self.word_embedding(inputs_ids[start_bdx:end_bdx])  # [1, seq_len, hidden_size]

            # 多GPU场景：模型参数分布在不同GPU上
            if self.num_gpus > 1:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    # 将hidden_states移动到下一层所在的GPU
                    hidden_states = self.parameter_move(hidden_states, ldx)
                    torch.cuda.empty_cache()
                # 将结果移动到第一个GPU
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :].to(self.layers[0].device)
            else:
                # 单GPU场景
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    torch.cuda.empty_cache()
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :]
        
        # 最终的归一化和语言模型头投影
        last_hidden_states = self.layernorm(last_hidden_states.contiguous(), self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(last_hidden_states)
        
        return logits
        

    def decode_forward(self, inputs_ids):
        """
        解码阶段的前向传播
        
        参数:
            inputs_ids: 新生成的token ID [batch_size, 1]
            
        返回:
            下一个token的logits [batch_size, 1, vocab_size]
            
        注意：解码阶段每次只处理一个新token，
        依赖之前预填充和解码阶段存储的KV缓存
        """
        hidden_states = self.word_embedding(inputs_ids)

        if self.num_gpus > 1:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
                hidden_states = self.parameter_move(hidden_states, ldx)
            hidden_states = hidden_states.to(self.layers[0].device)
        else:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
        
        # 只取最后一个位置的隐藏状态
        hidden_states = self.layernorm(hidden_states[:, -1:, :], self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(hidden_states)
        
        return logits


    def inference(self, inputs_ids):
        """
        完整的推理流程
        
        参数:
            inputs_ids: 输入序列 [batch_size, seq_len]
            
        返回:
            生成的token ID列表
            
        流程:
        1. 预填充阶段：处理整个输入序列
        2. 解码阶段：逐个生成新token
        
        性能指标：
        - 预填充延迟：处理输入序列的时间
        - 解码延迟：每生成一个token的平均时间
        - 吞吐量：每秒生成的token数
        """
        logger = get_logger()
        outputs_ids = []    # 存储所有生成的token
        output_ids = []     # 当前迭代生成的token
        
        logger.info("开始推理流程", 
                   input_shape=inputs_ids.shape,
                   max_new_length=self.max_new_length)
        
        # 预填充阶段
        start_phase("prefill")
        log_gpu_memory(phase="prefill_start")
        
        # 同步GPU确保时间测量准确
        torch.cuda.synchronize()
        prefill_start = time.time()

        # 预填充：处理整个输入序列
        logits = self.prefill_forward(inputs_ids=inputs_ids)
        output_ids = logits.argmax(dim=-1)
        outputs_ids.append(output_ids)
        # 执行必要的数据移动（例如将CPU缓存的数据传输到GPU）
        self.move()

        torch.cuda.synchronize()
        prefill_end = time.time()
        prefill_time = prefill_end - prefill_start
        
        log_gpu_memory(phase="prefill_end")
        end_phase("prefill")
        
        logger.info(f"预填充阶段完成", 
                   duration_s=round(prefill_time, 4),
                   tokens_processed=inputs_ids.shape[1])
        print(colored(f"Prefilling latency: {round(prefill_time, 4)} s\n", 'green'))

        # 解码阶段
        start_phase("decode")
        log_gpu_memory(phase="decode_start")
        
        decode_start = time.time()

        # 解码循环：逐个生成新token
        for step in range(self.max_new_length-1):
            if step % 10 == 0:  # 每10步记录一次进度
                logger.debug(f"解码进度: {step}/{self.max_new_length-1}")
                
            # 基于之前生成的token计算下一个token的概率分布
            logits = self.decode_forward(inputs_ids=output_ids)
            # 贪婪解码：选择概率最大的token
            output_ids = logits.argmax(dim=-1)
            outputs_ids.append(output_ids)

        decode_end = time.time()
        decode_time = decode_end - decode_start
        tokens_per_sec = self.batch_size * (self.max_new_length - 1) / decode_time
        ms_per_step = decode_time * 1000 / (self.max_new_length - 1)
        
        log_gpu_memory(phase="decode_end")
        end_phase("decode")
        
        # 记录推理统计信息
        stats = {
            "prefill_latency_s": prefill_time,
            "decode_latency_s": decode_time,
            "total_latency_s": prefill_time + decode_time,
            "decode_ms_per_step": ms_per_step,
            "throughput_tokens_per_s": tokens_per_sec,
            "input_tokens": inputs_ids.shape[1],
            "generated_tokens": self.max_new_length - 1,
            "batch_size": self.batch_size
        }
        
        log_inference_stats(stats)
        
        print(colored(
            f"Decoding latency: {round(ms_per_step, 2)} ms/step, "
            f"Throughput: {round(tokens_per_sec, 2)} tokens/s\n",
            'green'
        ))
        
        # 合并所有生成的token
        outputs_ids = torch.cat(outputs_ids, dim=-1).tolist()
        
        logger.info("推理完成", total_tokens_generated=len(outputs_ids) * len(outputs_ids[0]))
        
        return outputs_ids


    def generate(self, attention_type, inputs_ids, attention_masks, max_new_length, attn_config=None):
        """
        生成接口 - 模型推理的主入口
        
        参数:
            attention_type: 注意力类型（'flash_attn' 或 'retroinfer_attn'）
            inputs_ids: 输入token ID [batch_size, seq_len]
            attention_masks: 注意力掩码，标记哪些位置是有效的
            max_new_length: 最大生成长度
            attn_config: 注意力配置（RetroInfer特定参数）
            
        返回:
            生成的token ID序列
            
        重要：这是用户调用的主要接口，负责初始化缓存并启动推理流程
        """
        logger = get_logger()
        
        bs, input_length = inputs_ids.shape
        logger.info("开始生成任务", 
                   attention_type=attention_type,
                   batch_size=bs,
                   input_length=input_length,
                   max_new_length=max_new_length,
                   total_max_length=self.max_length)
        
        # 验证序列长度不超过模型的最大支持长度
        # 这是为了防止KV缓存溢出和位置编码越界
        assert input_length + max_new_length <= self.max_length, \
        f"Error: input_length({input_length}) + max_new_length({max_new_length}) exceeds max_length({self.max_length})"

        self.batch_size = bs
        self.input_length = input_length
        self.max_new_length = max_new_length
        self.attention_type = attention_type

        # 计算每个批次中有效token的起始位置
        # attention_masks中0表示padding，1表示有效token
        # valid_start记录了每个序列中第一个有效token的位置
        valid_start = attention_masks.shape[1] - torch.sum(attention_masks, dim=-1).detach().cpu().numpy()
        logger.debug("计算有效token起始位置", valid_start=valid_start.tolist())
        
        # 释放attention_masks以节省内存
        del attention_masks
        torch.cuda.empty_cache()

        logger.info("开始分配GPU缓冲区和CPU固定内存")
        start_phase("cache_initialization")
        
        # 初始化KV缓存
        # 根据attention_type选择不同的缓存策略：
        # - flash_attn: 全部缓存在GPU上
        # - retroinfer_attn: 使用GPU-CPU混合缓存
        self.init_kv_cache(input_length, valid_start, attn_config)
        
        end_phase("cache_initialization")
        
        # 执行推理
        outputs = self.inference(inputs_ids)
        
        logger.info("生成任务完成")

        return outputs


    def init_kv_cache(self, real_input_length, valid_start, attn_config=None):
        """
        初始化KV缓存（基类默认实现）
        
        参数:
            real_input_length: 实际输入长度
            valid_start: 有效序列的起始位置
            attn_config: 注意力配置
            
        这是基类的默认实现，支持Flash Attention缓存。
        具体模型可以重写此方法以支持特定的缓存类型。
        """
        logger = get_logger()
        logger.info("初始化标准Flash Attention缓存", 
                   attention_type=getattr(self, 'attention_type', 'unknown'),
                   input_length=real_input_length,
                   batch_size=getattr(self, 'batch_size', 'unknown'))
        
        try:
            # 导入缓存模块
            from cache_hub import flash_attn_cache
            
            # 默认使用Flash Attention缓存
            self.kv_cache = flash_attn_cache(
                valid_start=valid_start,
                layer_num=getattr(self, 'num_layers', 32),
                batch_size=getattr(self, 'batch_size', 1),
                max_length=getattr(self, 'max_new_length', 100) + real_input_length,
                num_key_value_heads=getattr(self, 'num_key_value_heads', 32),
                num_heads=getattr(self, 'num_heads', 32),
                head_dim=getattr(self, 'head_dim', 128),
                dtype=getattr(self, 'dtype', torch.float16),
                layer_mapping=getattr(self, 'layer_mapping', {}),
                num_gpus=getattr(self, 'num_gpus', 1)
            )
            logger.info("Flash Attention缓存初始化成功")
            
        except Exception as e:
            logger.error("KV缓存初始化失败", error=e)
            raise


    def move(self):
        """
        执行必要的数据移动操作
        
        某些缓存实现需要在预填充后执行数据移动
        （例如将数据从GPU移动到CPU）
        """
        if hasattr(self, 'kv_cache') and hasattr(self.kv_cache, 'move'):
            self.kv_cache.move()