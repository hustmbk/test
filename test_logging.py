# test_logging.py - 测试日志系统的功能
# 验证日志配置是否正常工作

import os
import sys
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_global_logger, get_logger, log_model_info, log_gpu_memory, start_phase, end_phase, log_inference_stats


def test_basic_logging():
    """测试基础日志功能"""
    print("=== 测试基础日志功能 ===")
    
    # 设置全局日志管理器
    logger = setup_global_logger(
        log_level="DEBUG",
        log_dir="logs",
        enable_file_logging=True,
        enable_console_logging=True
    )
    
    # 测试不同级别的日志
    logger.debug("这是一条调试信息")
    logger.info("这是一条普通信息")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    logger.critical("这是一条严重错误信息")
    
    print("✓ 基础日志功能测试完成\n")


def test_context_logging():
    """测试带上下文的日志"""
    print("=== 测试上下文日志功能 ===")
    
    logger = get_logger()
    
    # 测试带参数的日志
    logger.info("模型加载中", model_name="DeepSeek-V3", device="cuda:0", batch_size=4)
    logger.debug("处理批次", batch_idx=1, seq_len=128, hidden_size=4096)
    logger.warning("内存使用率较高", gpu_memory_gb=15.2, threshold_gb=16.0)
    
    # 测试错误日志
    try:
        raise ValueError("测试错误")
    except Exception as e:
        logger.error("捕获到异常", error=e, layer_idx=5, phase="prefill")
    
    print("✓ 上下文日志功能测试完成\n")


def test_model_info_logging():
    """测试模型信息日志"""
    print("=== 测试模型信息日志 ===")
    
    # 模拟模型信息
    model_info = {
        "model_version": "v3",
        "total_parameters": "671B",
        "active_parameters": "37B",
        "activation_ratio": "5.5%",
        "num_experts": 256,
        "experts_per_token": 8,
        "kv_compression_ratio": "32x",
        "max_context_length": 128000
    }
    
    log_model_info(model_info)
    print("✓ 模型信息日志测试完成\n")


def test_gpu_memory_logging():
    """测试GPU内存监控日志"""
    print("=== 测试GPU内存监控 ===")
    
    if torch.cuda.is_available():
        log_gpu_memory(device="cuda:0", phase="initialization")
        
        # 分配一些GPU内存
        dummy_tensor = torch.randn(1000, 1000).cuda()
        log_gpu_memory(device="cuda:0", phase="after_allocation")
        
        # 清理内存
        del dummy_tensor
        torch.cuda.empty_cache()
        log_gpu_memory(device="cuda:0", phase="after_cleanup")
    else:
        print("未检测到CUDA设备，跳过GPU内存监控测试")
    
    print("✓ GPU内存监控测试完成\n")


def test_phase_timing():
    """测试阶段计时功能"""
    print("=== 测试阶段计时功能 ===")
    
    import time
    
    # 模拟预填充阶段
    start_phase("prefill")
    time.sleep(0.1)  # 模拟处理时间
    end_phase("prefill")
    
    # 模拟解码阶段
    start_phase("decode")
    time.sleep(0.05)  # 模拟处理时间
    end_phase("decode")
    
    print("✓ 阶段计时功能测试完成\n")


def test_inference_stats():
    """测试推理统计日志"""
    print("=== 测试推理统计日志 ===")
    
    # 模拟推理统计数据
    stats = {
        "prefill_latency_s": 0.156,
        "decode_latency_s": 2.345,
        "total_latency_s": 2.501,
        "decode_ms_per_step": 23.45,
        "throughput_tokens_per_s": 42.6,
        "input_tokens": 1024,
        "generated_tokens": 100,
        "batch_size": 1
    }
    
    log_inference_stats(stats)
    print("✓ 推理统计日志测试完成\n")


def test_error_handling():
    """测试错误处理和异常日志"""
    print("=== 测试错误处理功能 ===")
    
    from utils.logger import log_error_with_context
    
    # 模拟不同类型的错误
    try:
        # 模拟内存不足错误
        raise RuntimeError("CUDA out of memory")
    except Exception as e:
        log_error_with_context(e, {
            "layer_idx": 15,
            "batch_size": 8,
            "seq_len": 4096,
            "phase": "attention_computation"
        })
    
    try:
        # 模拟配置错误
        raise ValueError("Invalid configuration parameter")
    except Exception as e:
        log_error_with_context(e, {
            "config_key": "num_attention_heads",
            "config_value": 0,
            "expected_range": "1-128"
        })
    
    print("✓ 错误处理功能测试完成\n")


def main():
    """运行所有日志测试"""
    print("开始日志系统测试\n")
    
    test_basic_logging()
    test_context_logging()
    test_model_info_logging()
    test_gpu_memory_logging()
    test_phase_timing()
    test_inference_stats()
    test_error_handling()
    
    print("=== 日志系统测试完成 ===")
    print("请查看 logs/ 目录下的日志文件以验证文件日志功能")


if __name__ == "__main__":
    main()