# moe_layer.py - DeepSeek MoE层实现
# 实现细粒度专家和共享专家隔离的MoE架构

import torch
import torch.nn.functional as F
import flashinfer
from typing import Tuple, Optional


class DeepSeekMoELayer:
    """
    DeepSeek MoE层实现
    
    特点：
    1. 细粒度专家分割
    2. 共享专家隔离
    3. 无辅助损失的负载均衡
    4. 支持多GPU专家并行
    """
    
    def __init__(self, layer_idx, config, device):
        """
        初始化MoE层
        
        参数:
            layer_idx: 层索引
            config: MoE配置
                - num_experts: 路由专家数量（如256）
                - num_shared_experts: 共享专家数量（通常为1）
                - top_k: 每个token激活的专家数（如8）
                - expert_hidden_dim: 专家隐藏维度
                - hidden_size: 模型隐藏维度
                - activation_fn: 激活函数类型
            device: 设备
        """
        self.layer_idx = layer_idx
        self.device = device
        self.config = config
        
        self.num_experts = config.num_experts
        self.num_shared_experts = config.num_shared_experts
        self.top_k = config.top_k
        self.expert_hidden_dim = config.expert_hidden_dim
        self.hidden_size = config.hidden_size
        
        # 路由器参数
        self.router = None  # 将在init_from_hf中初始化
        
        # 专家参数
        self.routed_experts = []  # 路由专家列表
        self.shared_experts = []  # 共享专家列表
        
        # 负载均衡统计
        self.expert_load_tracker = torch.zeros(self.num_experts, device=device)
        self.load_balance_bias = torch.zeros(self.num_experts, device=device)
        
    def init_from_hf(self, hf_moe_layer):
        """
        从HuggingFace格式的MoE层初始化
        
        参数:
            hf_moe_layer: HuggingFace的MoE层
        """
        # 初始化路由器
        self.router = hf_moe_layer.router.weight.detach().to(self.device, non_blocking=True)
        
        # 初始化路由专家
        for i in range(self.num_experts):
            expert = MoEExpert(
                gate_proj=hf_moe_layer.experts[i].gate_proj.weight.detach(),
                up_proj=hf_moe_layer.experts[i].up_proj.weight.detach(),
                down_proj=hf_moe_layer.experts[i].down_proj.weight.detach(),
                device=self.device
            )
            self.routed_experts.append(expert)
        
        # 初始化共享专家
        for i in range(self.num_shared_experts):
            shared_idx = self.num_experts + i
            expert = MoEExpert(
                gate_proj=hf_moe_layer.experts[shared_idx].gate_proj.weight.detach(),
                up_proj=hf_moe_layer.experts[shared_idx].up_proj.weight.detach(),
                down_proj=hf_moe_layer.experts[shared_idx].down_proj.weight.detach(),
                device=self.device
            )
            self.shared_experts.append(expert)
    
    def route(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算路由分数并选择专家
        
        参数:
            hidden_states: 输入隐藏状态 [batch_size, seq_len, hidden_size]
            
        返回:
            expert_indices: 选中的专家索引 [batch_size * seq_len, top_k]
            expert_weights: 专家权重 [batch_size * seq_len, top_k]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        # 计算路由分数
        router_logits = F.linear(hidden_states_flat, self.router)  # [batch_size * seq_len, num_experts]
        
        # 应用负载均衡偏置（不参与梯度计算）
        router_logits_balanced = router_logits + self.load_balance_bias.detach()
        
        # 选择top-k专家
        expert_weights, expert_indices = torch.topk(
            router_logits_balanced, self.top_k, dim=-1
        )
        
        # 归一化权重
        expert_weights = F.softmax(expert_weights, dim=-1)
        
        # 更新负载统计
        self._update_load_statistics(expert_indices)
        
        return expert_indices, expert_weights
    
    def _update_load_statistics(self, expert_indices: torch.Tensor):
        """
        更新专家负载统计和调整偏置
        
        参数:
            expert_indices: 选中的专家索引
        """
        # 统计每个专家被选中的次数
        for i in range(self.num_experts):
            count = (expert_indices == i).sum().float()
            self.expert_load_tracker[i] = 0.9 * self.expert_load_tracker[i] + 0.1 * count
        
        # 计算平均负载
        avg_load = self.expert_load_tracker.mean()
        
        # 调整偏置：过载的专家降低分数，欠载的专家提高分数
        load_ratio = self.expert_load_tracker / (avg_load + 1e-6)
        self.load_balance_bias = -0.01 * torch.log(load_ratio + 1e-6)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        MoE层前向传播
        
        参数:
            hidden_states: 输入隐藏状态 [batch_size, seq_len, hidden_size]
            
        返回:
            输出隐藏状态 [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 1. 计算共享专家输出
        shared_output = torch.zeros_like(hidden_states)
        for shared_expert in self.shared_experts:
            shared_output += shared_expert(hidden_states)
        
        # 2. 路由到专家
        expert_indices, expert_weights = self.route(hidden_states)
        
        # 3. 计算路由专家输出
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        routed_output = torch.zeros_like(hidden_states_flat)
        
        # 按专家分组处理以提高效率
        for expert_idx in range(self.num_experts):
            # 找到分配给该专家的token
            token_indices = (expert_indices == expert_idx).any(dim=-1).nonzero(as_tuple=True)[0]
            
            if token_indices.numel() > 0:
                # 获取这些token的输入
                expert_input = hidden_states_flat[token_indices]
                
                # 计算专家输出
                expert_output = self.routed_experts[expert_idx](expert_input.unsqueeze(1)).squeeze(1)
                
                # 获取权重
                weights_mask = (expert_indices[token_indices] == expert_idx)
                weights = expert_weights[token_indices][weights_mask].unsqueeze(-1)
                
                # 加权累加到输出
                routed_output[token_indices] += weights * expert_output
        
        routed_output = routed_output.view(batch_size, seq_len, hidden_size)
        
        # 4. 组合共享专家和路由专家的输出
        output = shared_output + routed_output
        
        return output


class MoEExpert:
    """
    单个MoE专家
    """
    
    def __init__(self, gate_proj, up_proj, down_proj, device):
        """
        初始化专家
        
        参数:
            gate_proj: Gate投影权重
            up_proj: Up投影权重
            down_proj: Down投影权重
            device: 设备
        """
        # 合并gate和up投影以优化计算
        self.gate_up_proj = torch.cat([gate_proj, up_proj], dim=0).to(device, non_blocking=True)
        self.down_proj = down_proj.to(device, non_blocking=True)
        
        del gate_proj, up_proj  # 释放内存
        
    def __call__(self, hidden_states):
        """
        专家前向传播
        
        参数:
            hidden_states: 输入隐藏状态
            
        返回:
            专家输出
        """
        # Gate和Up投影
        gate_up = F.linear(hidden_states, self.gate_up_proj)
        dim = gate_up.shape[-1] // 2
        
        # 分离gate和up
        gate = gate_up[..., :dim]
        up = gate_up[..., dim:]
        
        # SiLU(gate) * up
        hidden_states = F.silu(gate) * up
        
        # Down投影
        hidden_states = F.linear(hidden_states, self.down_proj)
        
        return hidden_states