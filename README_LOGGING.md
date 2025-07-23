# README_LOGGING.md - 日志系统使用指南

# RetrievalAttention 日志系统

## 概述

本项目集成了完整的日志系统，用于追踪模型加载、推理过程、性能指标和错误诊断。日志系统支持多级别输出、文件存储、GPU内存监控和阶段性能分析。

## 功能特性

### 🔍 多级别日志
- **DEBUG**: 详细的调试信息，包括每层处理详情
- **INFO**: 一般信息，模型加载、推理进度等
- **WARNING**: 警告信息，性能问题、回退操作等
- **ERROR**: 错误信息，异常和失败操作
- **CRITICAL**: 严重错误，系统无法继续运行

### 📊 性能监控
- GPU内存使用情况实时监控
- 推理阶段计时（预填充、解码）
- 吞吐量和延迟统计
- 缓存命中率和压缩比

### 🎨 输出格式
- 彩色控制台输出，易于区分日志级别
- 详细的文件日志，包含时间戳和调用位置
- 结构化日志信息，便于分析和过滤

### 💾 存储管理
- 自动日志文件轮换
- 可配置的最大文件数量
- 按时间戳命名的日志文件

## 快速开始

### 1. 基础使用

```python
from utils.logger import setup_global_logger, get_logger

# 初始化日志系统（在程序开始时调用一次）
logger = setup_global_logger(
    log_level="INFO",
    log_dir="logs",
    enable_file_logging=True,
    enable_console_logging=True
)

# 使用日志
logger = get_logger()
logger.info("程序开始运行")
logger.debug("详细调试信息", batch_size=4, seq_len=128)
logger.warning("内存使用率较高", gpu_memory_gb=15.2)
```

### 2. 配置化使用

```python
from logging_config import setup_logging

# 使用预定义配置
logger = setup_logging("production")  # 或 "development", "testing", "debug"

# 自定义配置
custom_config = {
    "log_level": "DEBUG",
    "log_dir": "custom_logs",
    "enable_file_logging": True,
    "enable_console_logging": False
}
logger = setup_logging(custom_config=custom_config)
```

### 3. 模型集成使用

```python
from model_hub.deepseek import DeepSeekModel
from logging_config import setup_logging

# 设置日志
setup_logging("performance")

# 初始化模型（会自动记录详细日志）
model = DeepSeekModel(
    model_name="deepseek-ai/DeepSeek-V2-Lite-Chat",
    max_length=4096,
    dtype=torch.float16,
    device_map="cuda:0",
    model_version="v2-lite"
)

# 推理（会自动记录性能指标）
outputs = model.generate(...)
```

## 预定义配置

### Development（开发模式）
```python
{
    "log_level": "DEBUG",
    "log_dir": "logs", 
    "enable_file_logging": True,
    "enable_console_logging": True,
    "max_log_files": 20
}
```

### Production（生产模式）
```python
{
    "log_level": "INFO",
    "log_dir": "logs",
    "enable_file_logging": True,
    "enable_console_logging": True, 
    "max_log_files": 100
}
```

### Performance（性能分析）
```python
{
    "log_level": "INFO",
    "log_dir": "performance_logs",
    "enable_file_logging": True,
    "enable_console_logging": True,
    "max_log_files": 50
}
```

### Testing（测试模式）
```python
{
    "log_level": "WARNING",
    "log_dir": "test_logs",
    "enable_file_logging": True,
    "enable_console_logging": False,
    "max_log_files": 5
}
```

## 环境变量配置

```bash
# 设置日志级别
export LOG_LEVEL=DEBUG

# 设置日志目录
export LOG_DIR=/path/to/logs

# 运行程序
python your_script.py
```

## 高级功能

### 性能监控

```python
from utils.logger import start_phase, end_phase, log_gpu_memory, log_inference_stats

# 阶段计时
start_phase("model_loading")
# ... 模型加载代码 ...
end_phase("model_loading")

# GPU内存监控
log_gpu_memory(device="cuda:0", phase="after_model_init")

# 推理统计
stats = {
    "prefill_latency_s": 0.156,
    "decode_latency_s": 2.345,
    "throughput_tokens_per_s": 42.6,
    "batch_size": 1
}
log_inference_stats(stats)
```

### 错误追踪

```python
from utils.logger import log_error_with_context

try:
    # 可能出错的代码
    model.forward(inputs)
except Exception as e:
    log_error_with_context(e, {
        "layer_idx": 15,
        "batch_size": 8,
        "seq_len": 4096,
        "phase": "attention_computation"
    })
    raise
```

### 模型信息记录

```python
from utils.logger import log_model_info

model_info = {
    "model_version": "v3",
    "total_parameters": "671B", 
    "active_parameters": "37B",
    "kv_compression_ratio": "32x",
    "max_context_length": 128000
}
log_model_info(model_info)
```

## 日志输出示例

### 控制台输出
```
2024-01-20 10:30:15 - RetrievalAttention - INFO - 开始初始化DeepSeek v3模型 [model_name=deepseek-ai/DeepSeek-V3, max_length=4096, dtype=torch.float16]
2024-01-20 10:30:16 - RetrievalAttention - INFO - 成功加载tokenizer和配置
2024-01-20 10:30:18 - RetrievalAttention - INFO - DeepSeek v3 模型参数 [num_layers=60, hidden_size=7168, num_experts=256, kv_compression=14.0x]
2024-01-20 10:30:20 - RetrievalAttention - INFO - 开始阶段: model_initialization
2024-01-20 10:30:22 - RetrievalAttention - INFO - GPU内存使用 - 已分配: 12.34GB, 已保留: 13.56GB [device=cuda:0]
2024-01-20 10:30:25 - RetrievalAttention - INFO - 完成阶段: model_initialization, 耗时: 5.2341s
```

### 文件日志
```
2024-01-20 10:30:15 - RetrievalAttention - DEBUG - __init__:55 - 初始化MLAAttention层 0 [device=cuda:0]
2024-01-20 10:30:15 - RetrievalAttention - INFO - __init__:86 - MLAAttention层 0 配置完成 [hidden_size=7168, num_heads=56, head_dim=128, q_lora_rank=1536, kv_lora_rank=512, compression_ratio=14.0x]
2024-01-20 10:30:15 - RetrievalAttention - DEBUG - __init__:96 - MLAAttention层 0 投影矩阵初始化完成
```

## 性能影响

### 最小化性能开销
- 使用延迟格式化，只在需要时生成日志字符串
- 可配置的日志级别，生产环境可关闭DEBUG日志
- 异步文件写入，不阻塞主线程
- 高效的内存管理，避免日志导致的内存泄漏

### 推荐设置
- **开发**: DEBUG级别，便于调试
- **测试**: WARNING级别，只记录问题
- **生产**: INFO级别，平衡信息量和性能
- **性能分析**: INFO级别 + 详细统计

## 故障排除

### 常见问题

1. **日志文件未生成**
   - 检查目录权限
   - 确认enable_file_logging=True
   - 查看控制台是否有错误信息

2. **日志级别过低**
   - 检查全局日志级别设置
   - 确认环境变量LOG_LEVEL
   - 使用get_logger().setLevel()动态调整

3. **GPU内存监控失败**
   - 确认CUDA可用性
   - 检查设备名称是否正确
   - 查看PyTorch版本兼容性

### 调试技巧

```python
# 临时提高日志级别
from utils.logger import get_logger
logger = get_logger()
logger.logger.setLevel(logging.DEBUG)

# 检查当前配置
print(f"当前日志级别: {logger.logger.level}")
print(f"处理器数量: {len(logger.logger.handlers)}")

# 强制刷新日志
for handler in logger.logger.handlers:
    handler.flush()
```

## 最佳实践

1. **在程序开始时初始化日志系统**
2. **使用合适的日志级别**
3. **添加有用的上下文信息**
4. **定期清理旧日志文件**
5. **在异常处理中记录详细信息**
6. **使用阶段计时监控性能瓶颈**
7. **监控GPU内存使用避免OOM**

## 扩展开发

如需扩展日志系统功能，可以：

1. **添加新的日志处理器**（如网络日志、数据库日志）
2. **实现自定义格式化器**（如JSON格式）
3. **集成监控系统**（如Prometheus、Grafana）
4. **添加日志分析工具**（如ELK Stack）

参考`utils/logger.py`了解详细实现。