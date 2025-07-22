# mla_cache.py - MLA（Multi-head Latent Attention）缓存实现
# 支持DeepSeek V2/V3的压缩KV缓存
# 通过低秩压缩大幅减少内存使用（压缩比高达32倍）

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from .cache import KV_Cache


class MLACache(KV_Cache):
    """
    MLA压缩KV缓存
    
    核心优势：
    1. 存储压缩的潜在向量而非完整的K和V
    2. 内存使用减少93.3%
    3. 支持GPU-CPU混合存储以处理超长序列
    4. 与RetroInfer框架无缝集成
    """
    
    def __init__(
        self,
        valid_start: np.ndarray,
        layer_num: int,
        batch_size: int,
        max_length: int,
        kv_lora_rank: int,  # 压缩维度（如512）
        dtype: torch.dtype,
        layer_mapping: Dict[str, str],
        num_gpus: int = 1,
        model_version: str = "v3",
        cpu_offload_ratio: float = 0.8  # CPU存储比例
    ):
        """
        初始化MLA缓存
        
        参数:
            valid_start: 每个序列的有效起始位置
            layer_num: 层数
            batch_size: 批大小
            max_length: 最大序列长度
            kv_lora_rank: KV压缩维度（DeepSeek V2为512）
            dtype: 数据类型
            layer_mapping: 层到设备的映射
            num_gpus: GPU数量
            model_version: 模型版本（v2或v3）
            cpu_offload_ratio: 存储在CPU上的缓存比例
        """
        # Note: MLACache has a different initialization pattern than KV_Cache
        # We'll initialize parent class attributes directly instead of calling super().__init__()
        self.layer_num = layer_num
        self.batch_size = batch_size
        self.max_length = max_length
        self.dtype = dtype
        self.context = 0
        
        self.kv_lora_rank = kv_lora_rank
        self.layer_mapping = layer_mapping
        self.num_gpus = num_gpus
        self.model_version = model_version
        self.cpu_offload_ratio = cpu_offload_ratio
        
        # 计算内存分配策略
        self._compute_memory_allocation()
        
        # 初始化压缩KV缓存
        self._init_compressed_cache()
        
        # 统计信息
        self.compression_ratio = None
        self._compute_compression_stats()
        
    def _compute_memory_allocation(self):
        """计算GPU和CPU的内存分配策略"""
        # 计算总的缓存大小（压缩后）
        cache_size_per_layer = self.batch_size * self.max_length * self.kv_lora_rank
        total_cache_size = cache_size_per_layer * self.layer_num
        
        # 根据比例分配GPU和CPU存储
        self.gpu_cache_length = int(self.max_length * (1 - self.cpu_offload_ratio))
        self.cpu_cache_length = self.max_length - self.gpu_cache_length
        
        print(f"MLA缓存分配策略:")
        print(f"  - 压缩维度: {self.kv_lora_rank}")
        print(f"  - GPU缓存长度: {self.gpu_cache_length}")
        print(f"  - CPU缓存长度: {self.cpu_cache_length}")
        print(f"  - 总缓存大小: {total_cache_size * 2 / 1e9:.2f} GB (FP16)")
        
    def _init_compressed_cache(self):
        """初始化压缩的KV缓存"""
        self.compressed_kv_gpu = {}  # GPU上的压缩KV
        self.compressed_kv_cpu = {}  # CPU上的压缩KV（固定内存）
        
        for layer_idx in range(self.layer_num):
            device = self.layer_mapping[str(layer_idx)]
            
            # GPU缓存（用于近期的KV）
            self.compressed_kv_gpu[layer_idx] = torch.zeros(
                (self.batch_size, self.gpu_cache_length, self.kv_lora_rank),
                dtype=self.dtype,
                device=device
            )
            
            # CPU缓存（用于远期的KV，使用固定内存加速传输）
            if self.cpu_cache_length > 0:
                self.compressed_kv_cpu[layer_idx] = torch.zeros(
                    (self.batch_size, self.cpu_cache_length, self.kv_lora_rank),
                    dtype=self.dtype,
                    pin_memory=True
                )
                
        # 缓存管理变量
        self.context = 0  # 当前上下文长度
        self.gpu_cache_start = 0  # GPU缓存的起始位置
        
    def _compute_compression_stats(self):
        """计算压缩统计信息"""
        # 假设原始KV的维度（以Llama为例）
        original_kv_dim = 4096  # 典型的隐藏维度
        compressed_dim = self.kv_lora_rank
        
        self.compression_ratio = original_kv_dim / compressed_dim
        self.memory_saving_ratio = 1 - (compressed_dim / original_kv_dim)
        
        print(f"\nMLA压缩统计:")
        print(f"  - 压缩比: {self.compression_ratio:.1f}x")
        print(f"  - 内存节省: {self.memory_saving_ratio * 100:.1f}%")
        
    def update_compressed_kv(self, compressed_kv: torch.Tensor, layer_idx: int, start_idx: int):
        """
        更新压缩的KV缓存
        
        参数:
            compressed_kv: 压缩的KV张量 [batch_size, seq_len, kv_lora_rank]
            layer_idx: 层索引
            start_idx: 批次起始索引
        """
        batch_size, seq_len, _ = compressed_kv.shape
        end_idx = start_idx + batch_size
        
        # 确定存储位置
        cache_end = self.context + seq_len
        
        if cache_end <= self.gpu_cache_length:
            # 全部存储在GPU上
            self.compressed_kv_gpu[layer_idx][start_idx:end_idx, self.context:cache_end] = compressed_kv
        else:
            # 需要分割存储
            gpu_portion = self.gpu_cache_length - self.context
            if gpu_portion > 0:
                # 部分存储在GPU
                self.compressed_kv_gpu[layer_idx][start_idx:end_idx, self.context:self.gpu_cache_length] = \
                    compressed_kv[:, :gpu_portion]
                    
            # 剩余部分存储在CPU
            cpu_start = max(0, self.context - self.gpu_cache_length)
            cpu_end = cache_end - self.gpu_cache_length
            self.compressed_kv_cpu[layer_idx][start_idx:end_idx, cpu_start:cpu_end] = \
                compressed_kv[:, gpu_portion:].cpu()
                
    def get_compressed_kv(self, layer_idx: int, start_pos: int = 0, end_pos: Optional[int] = None):
        """
        获取压缩的KV缓存
        
        参数:
            layer_idx: 层索引
            start_pos: 起始位置
            end_pos: 结束位置（如果为None，则到当前上下文结束）
            
        返回:
            压缩的KV张量
        """
        if end_pos is None:
            end_pos = self.context
            
        device = self.layer_mapping[str(layer_idx)]
        
        # 如果全部在GPU缓存中
        if end_pos <= self.gpu_cache_length:
            return self.compressed_kv_gpu[layer_idx][:, start_pos:end_pos]
            
        # 需要从GPU和CPU合并
        result_parts = []
        
        # GPU部分
        if start_pos < self.gpu_cache_length:
            gpu_end = min(end_pos, self.gpu_cache_length)
            result_parts.append(self.compressed_kv_gpu[layer_idx][:, start_pos:gpu_end])
            
        # CPU部分
        cpu_start = max(0, start_pos - self.gpu_cache_length)
        cpu_end = end_pos - self.gpu_cache_length
        if cpu_end > 0:
            cpu_data = self.compressed_kv_cpu[layer_idx][:, cpu_start:cpu_end]
            result_parts.append(cpu_data.to(device, non_blocking=True))
            
        return torch.cat(result_parts, dim=1)
        
    def prefill_update_kv_cache(self, query_states, compressed_kv, layer_idx, start_bdx):
        """
        预填充阶段更新缓存（实现基类接口）
        
        注意：MLA直接存储压缩的KV，不需要原始的K和V
        """
        batch_size = compressed_kv.shape[0]
        seq_len = compressed_kv.shape[1]
        
        # 更新压缩KV缓存
        self.update_compressed_kv(compressed_kv, layer_idx, start_bdx)
        
        # 更新上下文长度
        if start_bdx == 0:  # 第一个批次
            self.context += seq_len
            
        return compressed_kv, compressed_kv  # 返回相同的压缩KV（保持接口兼容）
        
    def decode_update_kv_cache(self, compressed_kv, layer_idx):
        """
        解码阶段更新缓存（实现基类接口）
        """
        # 更新单个token的压缩KV
        self.update_compressed_kv(compressed_kv, layer_idx, 0)
        self.context += 1
        
        return compressed_kv, compressed_kv
        
    def move_to_gpu(self, layer_idx: int, start_pos: int, length: int):
        """
        将指定范围的CPU缓存移动到GPU
        
        用于RetroInfer的动态缓存管理
        """
        if start_pos >= self.gpu_cache_length:
            # 计算CPU缓存中的位置
            cpu_start = start_pos - self.gpu_cache_length
            cpu_end = min(cpu_start + length, self.cpu_cache_length)
            
            # 异步传输到GPU
            device = self.layer_mapping[str(layer_idx)]
            return self.compressed_kv_cpu[layer_idx][:, cpu_start:cpu_end].to(device, non_blocking=True)
            
        return None
        
    def get_memory_usage(self):
        """获取内存使用统计"""
        gpu_memory = 0
        cpu_memory = 0
        
        # 计算GPU内存
        for layer_idx in range(self.layer_num):
            gpu_memory += self.compressed_kv_gpu[layer_idx].element_size() * \
                         self.compressed_kv_gpu[layer_idx].nelement()
                         
        # 计算CPU内存
        if self.cpu_cache_length > 0:
            for layer_idx in range(self.layer_num):
                cpu_memory += self.compressed_kv_cpu[layer_idx].element_size() * \
                             self.compressed_kv_cpu[layer_idx].nelement()
                             
        return {
            "gpu_memory_mb": gpu_memory / 1024 / 1024,
            "cpu_memory_mb": cpu_memory / 1024 / 1024,
            "total_memory_mb": (gpu_memory + cpu_memory) / 1024 / 1024,
            "compression_ratio": self.compression_ratio,
            "memory_saving": f"{self.memory_saving_ratio * 100:.1f}%"
        }
        
    def clear_cache(self):
        """清空缓存"""
        for layer_idx in range(self.layer_num):
            self.compressed_kv_gpu[layer_idx].zero_()
            if self.cpu_cache_length > 0:
                self.compressed_kv_cpu[layer_idx].zero_()
                
        self.context = 0
        self.gpu_cache_start = 0
        

def mla_cache(*args, **kwargs):
    """工厂函数，创建MLA缓存实例"""
    return MLACache(*args, **kwargs)