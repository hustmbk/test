# example_with_logging.py - 演示如何在现有模型中使用日志系统
# 基于 deepseek_simple_example.py 修改，添加了完整的日志支持

import sys
import os
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 首先设置日志系统
from utils.logger import setup_global_logger, get_logger, log_model_info

def main():
    """演示带日志的模型使用"""
    
    # 1. 初始化日志系统 (这应该在最开始调用)
    logger = setup_global_logger(
        log_level="INFO",  # 可以设置为 DEBUG, INFO, WARNING, ERROR, CRITICAL
        log_dir="logs",
        enable_file_logging=True,
        enable_console_logging=True
    )
    
    logger.info("=== 开始RetrievalAttention示例程序 ===")
    
    try:
        # 2. 导入模型类 (现在会自动使用日志系统)
        from model_hub.deepseek import DeepSeekModel
        
        # 3. 模型配置
        model_config = {
            "model_name": "deepseek-ai/DeepSeek-V2-Lite-Chat",  # 或者使用本地路径
            "max_length": 4096,
            "dtype": torch.float16,
            "device_map": "cuda:0" if torch.cuda.is_available() else "cpu",
            "model_version": "v2-lite"
        }
        
        logger.info("开始初始化DeepSeek模型", **model_config)
        
        # 4. 初始化模型 (会自动记录初始化过程)
        model = DeepSeekModel(
            model_name=model_config["model_name"],
            max_length=model_config["max_length"],
            dtype=model_config["dtype"],
            device_map=model_config["device_map"],
            model_version=model_config["model_version"]
        )
        
        # 5. 记录模型信息
        model_info = model.get_model_info()
        log_model_info(model_info)
        
        # 6. 准备测试输入
        test_text = "人工智能的发展历程可以分为以下几个阶段："
        
        logger.info("开始tokenize输入文本", input_text=test_text)
        inputs = model.tokenizer(test_text, return_tensors="pt")
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        logger.info("Tokenize完成", 
                   input_length=input_ids.shape[1],
                   tokens=input_ids.tolist())
        
        # 7. 执行推理 (会自动记录推理过程和性能指标)
        max_new_tokens = 50
        
        logger.info("开始模型推理", max_new_tokens=max_new_tokens)
        
        output_ids = model.generate(
            attention_type="MLA",  # 使用MLA注意力
            inputs_ids=input_ids,
            attention_masks=attention_mask,
            max_new_length=max_new_tokens
        )
        
        # 8. 解码输出
        logger.info("开始解码输出")
        generated_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # 9. 显示结果
        print("\n" + "="*50)
        print("生成结果:")
        print("="*50)
        print(generated_text)
        print("="*50)
        
        logger.info("推理完成", 
                   output_length=len(output_ids[0]),
                   generated_text_length=len(generated_text))
        
    except Exception as e:
        logger.error("程序执行过程中发生错误", error=e)
        raise
    
    finally:
        logger.info("=== 示例程序结束 ===")


if __name__ == "__main__":
    main()