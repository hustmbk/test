# logging_config.py - 日志配置文件
# 提供不同场景下的日志配置预设

import os
from utils.logger import setup_global_logger

# 预定义的日志配置
LOGGING_CONFIGS = {
    "development": {
        "log_level": "DEBUG",
        "log_dir": "logs",
        "enable_file_logging": True,
        "enable_console_logging": True,
        "max_log_files": 20
    },
    
    "production": {
        "log_level": "INFO", 
        "log_dir": "logs",
        "enable_file_logging": True,
        "enable_console_logging": True,
        "max_log_files": 100
    },
    
    "testing": {
        "log_level": "WARNING",
        "log_dir": "test_logs",
        "enable_file_logging": True, 
        "enable_console_logging": False,
        "max_log_files": 5
    },
    
    "performance": {
        "log_level": "INFO",
        "log_dir": "performance_logs",
        "enable_file_logging": True,
        "enable_console_logging": True,
        "max_log_files": 50
    },
    
    "debug": {
        "log_level": "DEBUG",
        "log_dir": "debug_logs", 
        "enable_file_logging": True,
        "enable_console_logging": True,
        "max_log_files": 10
    }
}


def setup_logging(config_name="development", custom_config=None):
    """
    根据配置名称或自定义配置设置日志系统
    
    参数:
        config_name: 预定义配置名称 ("development", "production", "testing", "performance", "debug")
        custom_config: 自定义配置字典，会覆盖预定义配置
        
    返回:
        配置好的日志管理器
    """
    if custom_config:
        config = custom_config
    elif config_name in LOGGING_CONFIGS:
        config = LOGGING_CONFIGS[config_name].copy()
    else:
        raise ValueError(f"未知的日志配置: {config_name}. 可用配置: {list(LOGGING_CONFIGS.keys())}")
    
    # 从环境变量读取配置覆盖
    config["log_level"] = os.getenv("LOG_LEVEL", config["log_level"])
    config["log_dir"] = os.getenv("LOG_DIR", config["log_dir"])
    
    # 设置日志管理器
    logger = setup_global_logger(**config)
    
    logger.info(f"日志系统已初始化", 
               config_name=config_name,
               log_level=config["log_level"],
               log_dir=config["log_dir"],
               file_logging=config["enable_file_logging"],
               console_logging=config["enable_console_logging"])
    
    return logger


def get_config_for_model(model_name):
    """
    根据模型名称推荐日志配置
    
    参数:
        model_name: 模型名称
        
    返回:
        推荐的配置名称
    """
    model_name = model_name.lower()
    
    # 大模型推荐详细日志
    if any(name in model_name for name in ["deepseek", "llama", "qwen", "chatglm"]):
        return "performance"
    
    # 测试模型使用调试配置
    if "test" in model_name or "demo" in model_name:
        return "debug"
    
    # 默认开发配置
    return "development"


# 环境变量配置说明
ENV_CONFIG_HELP = """
环境变量配置:
    LOG_LEVEL: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    LOG_DIR: 日志目录路径
    
示例:
    export LOG_LEVEL=DEBUG
    export LOG_DIR=/path/to/logs
    python your_script.py
"""

if __name__ == "__main__":
    print("可用的日志配置:")
    for name, config in LOGGING_CONFIGS.items():
        print(f"  {name}: {config}")
    
    print("\n" + ENV_CONFIG_HELP)