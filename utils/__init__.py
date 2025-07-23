# utils/__init__.py - 工具模块初始化
# 导出常用的工具函数和类

from .logger import (
    LLMLogger,
    setup_global_logger,
    get_logger,
    debug,
    info,
    warning,
    error,
    critical,
    log_model_info,
    log_gpu_memory,
    start_phase,
    end_phase,
    log_inference_stats,
    log_error_with_context
)

__all__ = [
    'LLMLogger',
    'setup_global_logger', 
    'get_logger',
    'debug',
    'info',
    'warning',
    'error',
    'critical',
    'log_model_info',
    'log_gpu_memory',
    'start_phase',
    'end_phase',
    'log_inference_stats',
    'log_error_with_context'
]