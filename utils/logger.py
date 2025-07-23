# logger.py - 统一日志配置模块
# 提供项目的集中式日志管理，支持多级别日志、文件输出和控制台输出

import os
import sys
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m'  # 紫色
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


class LLMLogger:
    """
    LLM推理框架的专用日志管理器
    
    功能特性:
    1. 多级别日志输出 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    2. 同时支持控制台和文件输出
    3. 性能监控日志
    4. GPU内存使用监控
    5. 模型推理阶段标记
    6. 自动日志文件轮换
    """
    
    def __init__(
        self,
        name: str = "RetrievalAttention",
        log_level: str = "INFO",
        log_dir: str = "logs",
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        max_log_files: int = 10
    ):
        """
        初始化日志管理器
        
        参数:
            name: 日志器名称
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: 日志文件目录
            enable_file_logging: 是否启用文件日志
            enable_console_logging: 是否启用控制台日志
            max_log_files: 最大日志文件数量
        """
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = log_dir
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.max_log_files = max_log_files
        
        # 创建日志目录
        if self.enable_file_logging and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # 初始化日志器
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.log_level)
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            self._setup_handlers()
        
        # 性能监控相关
        self.phase_timers = {}  # 阶段计时器
        self.memory_stats = {}  # 内存统计
        
    def _setup_handlers(self):
        """设置日志处理器"""
        
        # 控制台处理器
        if self.enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            
            # 彩色格式化
            console_formatter = ColoredFormatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # 文件处理器
        if self.enable_file_logging:
            # 生成日志文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = os.path.join(self.log_dir, f"{self.name}_{timestamp}.log")
            
            file_handler = logging.FileHandler(log_filename, encoding='utf-8')
            file_handler.setLevel(self.log_level)
            
            # 文件格式化（无颜色）
            file_formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            # 清理旧日志文件
            self._cleanup_old_logs()
    
    def _cleanup_old_logs(self):
        """清理旧的日志文件"""
        if not os.path.exists(self.log_dir):
            return
            
        log_files = [f for f in os.listdir(self.log_dir) if f.endswith('.log')]
        log_files.sort(key=lambda x: os.path.getctime(os.path.join(self.log_dir, x)))
        
        # 删除多余的日志文件
        if len(log_files) > self.max_log_files:
            for old_file in log_files[:-self.max_log_files]:
                try:
                    os.remove(os.path.join(self.log_dir, old_file))
                    self.logger.debug(f"已删除旧日志文件: {old_file}")
                except Exception as e:
                    self.logger.warning(f"删除日志文件失败: {old_file}, 错误: {e}")
    
    def debug(self, message: str, **kwargs):
        """记录调试信息"""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """记录一般信息"""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """记录警告信息"""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """记录错误信息"""
        if error:
            message = f"{message} - 错误详情: {str(error)}"
        self.logger.error(self._format_message(message, **kwargs))
    
    def critical(self, message: str, error: Optional[Exception] = None, **kwargs):
        """记录严重错误"""
        if error:
            message = f"{message} - 错误详情: {str(error)}"
        self.logger.critical(self._format_message(message, **kwargs))
    
    def _format_message(self, message: str, **kwargs) -> str:
        """格式化日志消息"""
        if kwargs:
            # 添加上下文信息
            context = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            return f"{message} [{context}]"
        return message
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """记录模型信息"""
        self.info("模型配置信息:")
        for key, value in model_info.items():
            self.info(f"  {key}: {value}")
    
    def log_gpu_memory(self, device: str = "cuda:0", phase: str = ""):
        """记录GPU内存使用情况"""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
                
                phase_str = f"[{phase}] " if phase else ""
                self.info(f"{phase_str}GPU内存使用 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB", 
                         device=device)
                
                # 存储内存统计
                self.memory_stats[phase or "unknown"] = {
                    "allocated_gb": allocated,
                    "reserved_gb": reserved,
                    "timestamp": time.time()
                }
        except Exception as e:
            self.warning(f"获取GPU内存信息失败: {e}")
    
    def start_phase(self, phase_name: str):
        """开始一个新的处理阶段"""
        self.phase_timers[phase_name] = time.time()
        self.info(f"开始阶段: {phase_name}")
        self.log_gpu_memory(phase=phase_name)
    
    def end_phase(self, phase_name: str):
        """结束一个处理阶段"""
        if phase_name in self.phase_timers:
            elapsed = time.time() - self.phase_timers[phase_name]
            self.info(f"完成阶段: {phase_name}, 耗时: {elapsed:.4f}s")
            del self.phase_timers[phase_name]
        else:
            self.warning(f"阶段 {phase_name} 未找到开始时间")
        
        self.log_gpu_memory(phase=f"{phase_name}_end")
    
    def log_inference_stats(self, stats: Dict[str, Any]):
        """记录推理统计信息"""
        self.info("推理性能统计:")
        for key, value in stats.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """记录带上下文的错误信息"""
        self.error(f"发生错误: {type(error).__name__}", error=error, **context)
    
    def log_cache_stats(self, layer_idx: int, cache_type: str, **stats):
        """记录缓存统计信息"""
        self.debug(f"缓存统计 - 层{layer_idx}({cache_type})", **stats)


# 全局日志管理器实例
_global_logger: Optional[LLMLogger] = None


def setup_global_logger(
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_file_logging: bool = True,
    enable_console_logging: bool = True
) -> LLMLogger:
    """
    设置全局日志管理器
    
    这应该在应用程序启动时调用一次
    """
    global _global_logger
    _global_logger = LLMLogger(
        log_level=log_level,
        log_dir=log_dir,
        enable_file_logging=enable_file_logging,
        enable_console_logging=enable_console_logging
    )
    return _global_logger


def get_logger() -> LLMLogger:
    """
    获取全局日志管理器实例
    
    如果还没有初始化，则使用默认配置创建
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_global_logger()
    return _global_logger


# 便捷函数
def debug(message: str, **kwargs):
    """记录调试信息"""
    get_logger().debug(message, **kwargs)


def info(message: str, **kwargs):
    """记录一般信息"""
    get_logger().info(message, **kwargs)


def warning(message: str, **kwargs):
    """记录警告信息"""
    get_logger().warning(message, **kwargs)


def error(message: str, error: Optional[Exception] = None, **kwargs):
    """记录错误信息"""
    get_logger().error(message, error, **kwargs)


def critical(message: str, error: Optional[Exception] = None, **kwargs):
    """记录严重错误"""
    get_logger().critical(message, error, **kwargs)


def log_model_info(model_info: Dict[str, Any]):
    """记录模型信息"""
    get_logger().log_model_info(model_info)


def log_gpu_memory(device: str = "cuda:0", phase: str = ""):
    """记录GPU内存使用情况"""
    get_logger().log_gpu_memory(device, phase)


def start_phase(phase_name: str):
    """开始一个新的处理阶段"""
    get_logger().start_phase(phase_name) 


def end_phase(phase_name: str):
    """结束一个处理阶段"""
    get_logger().end_phase(phase_name)


def log_inference_stats(stats: Dict[str, Any]):
    """记录推理统计信息"""
    get_logger().log_inference_stats(stats)


def log_error_with_context(error: Exception, context: Dict[str, Any]):
    """记录带上下文的错误信息"""
    get_logger().log_error_with_context(error, context)