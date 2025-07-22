# 使用本地模型路径指南

## 问题描述
当使用本地路径加载 DeepSeek 模型时，transformers 库可能会尝试将路径拼接为下载地址。

## 解决方案

### 1. 使用绝对路径并设置环境变量
```bash
# 禁用 transformers 的在线功能
export TRANSFORMERS_OFFLINE=1

# 使用绝对路径运行
python deepseek_simple_example.py --model-path /absolute/path/to/your/deepseek-v2-lite --model-version v2-lite
```

### 2. 修改代码支持本地路径
代码已经更新，现在支持 `--model-path` 参数：
```bash
python deepseek_simple_example.py --model-path ./models/deepseek-v2-lite --model-version v2-lite
```

### 3. 确保模型文件完整
本地模型目录应包含以下文件：
- `config.json` - 模型配置
- `tokenizer.json` - 分词器配置
- `tokenizer_config.json` - 分词器额外配置
- `pytorch_model.bin` 或 `model.safetensors` - 模型权重
- `special_tokens_map.json` - 特殊标记映射

### 4. 如果仍有问题，可以尝试：
```python
# 在代码中添加
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
```

### 5. 创建符号链接（可选）
如果模型名称格式有要求，可以创建符号链接：
```bash
ln -s /path/to/your/local/model /tmp/deepseek-ai/DeepSeek-V2-Lite
```

## 测试本地加载
```python
from transformers import AutoTokenizer, AutoConfig
import os

# 测试本地路径
local_path = "/path/to/your/model"
if os.path.exists(local_path):
    print(f"Loading from local path: {local_path}")
    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    config = AutoConfig.from_pretrained(local_path, local_files_only=True)
```

## 注意事项
- 使用 `local_files_only=True` 参数强制只使用本地文件
- 确保配置文件中的路径与实际路径一致
- 检查 `config/DeepSeek-V3.json` 中是否有硬编码的路径