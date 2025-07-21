#!/usr/bin/env python3
"""
Comprehensive DeepSeek-V3 + RetroInfer Integration Test Suite

This script provides a complete testing framework for evaluating DeepSeek models
with RetroInfer optimization, including performance benchmarks, memory profiling,
and quality assessment.
"""

import os
import sys
import json
import time
import torch
import argparse
import psutil
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored
from transformers import AutoTokenizer
import nvidia_ml_py as nvml

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from model_hub import DeepSeekModel


@dataclass
class TestResult:
    """Container for test results"""
    method: str
    model_version: str
    input_tokens: int
    output_tokens: int
    inference_time: float
    tokens_per_second: float
    gpu_memory_used: float
    cpu_memory_used: float
    peak_gpu_memory: float
    first_token_latency: float
    generated_text: str
    retroinfer_params: Optional[Dict] = None
    quality_scores: Optional[Dict] = None


class MemoryProfiler:
    """GPU and CPU memory profiling utility"""
    
    def __init__(self):
        nvml.nvmlInit()
        self.gpu_count = nvml.nvmlDeviceGetCount()
        self.handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.gpu_count)]
        
    def get_gpu_memory(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        total_used = 0
        total_free = 0
        total_capacity = 0
        
        for handle in self.handles:
            info = nvml.nvmlDeviceGetMemoryInfo(handle)
            total_used += info.used
            total_free += info.free
            total_capacity += info.total
            
        return {
            "used_gb": total_used / 1024**3,
            "free_gb": total_free / 1024**3,
            "total_gb": total_capacity / 1024**3,
            "utilization": total_used / total_capacity
        }
    
    def get_cpu_memory(self) -> Dict[str, float]:
        """Get current CPU memory usage"""
        mem = psutil.virtual_memory()
        return {
            "used_gb": mem.used / 1024**3,
            "available_gb": mem.available / 1024**3,
            "total_gb": mem.total / 1024**3,
            "percent": mem.percent
        }
    
    def profile_memory_during_inference(self, func, *args, **kwargs):
        """Profile memory usage during function execution"""
        memory_samples = []
        start_gpu = self.get_gpu_memory()
        start_cpu = self.get_cpu_memory()
        
        # Run function and collect memory samples
        import threading
        stop_profiling = threading.Event()
        
        def sample_memory():
            while not stop_profiling.is_set():
                memory_samples.append({
                    "gpu": self.get_gpu_memory(),
                    "cpu": self.get_cpu_memory(),
                    "timestamp": time.time()
                })
                time.sleep(0.1)  # Sample every 100ms
        
        profiler_thread = threading.Thread(target=sample_memory)
        profiler_thread.start()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Stop profiling
        stop_profiling.set()
        profiler_thread.join()
        
        end_gpu = self.get_gpu_memory()
        end_cpu = self.get_cpu_memory()
        
        # Calculate peak memory
        peak_gpu = max(sample["gpu"]["used_gb"] for sample in memory_samples)
        peak_cpu = max(sample["cpu"]["used_gb"] for sample in memory_samples)
        
        return result, {
            "gpu_delta": end_gpu["used_gb"] - start_gpu["used_gb"],
            "cpu_delta": end_cpu["used_gb"] - start_cpu["used_gb"],
            "peak_gpu": peak_gpu,
            "peak_cpu": peak_cpu,
            "samples": memory_samples
        }


class TestDataGenerator:
    """Generate various test scenarios"""
    
    @staticmethod
    def create_needle_in_haystack(context_length: int, needle_position: float = 0.5) -> Tuple[str, str]:
        """Create needle-in-haystack test"""
        # Base context
        context = """
        人工智能的发展历程充满了突破和创新。从早期的专家系统到现代的深度学习，
        每一步都代表着人类对智能本质理解的深化。深度学习特别是Transformer架构的出现，
        彻底改变了自然语言处理的格局。
        """
        
        # The needle (important information)
        needle = "关键信息：DeepSeek-V3的MLA机制将KV缓存压缩了32倍，同时保持了模型性能。"
        
        # Calculate positions
        chars_per_token = 4  # Approximate for Chinese
        total_chars = context_length * chars_per_token
        needle_pos = int(total_chars * needle_position)
        
        # Build full text
        pre_context = context * (needle_pos // len(context))
        post_context = context * ((total_chars - needle_pos - len(needle)) // len(context))
        
        full_text = pre_context[:needle_pos] + needle + post_context
        question = "\n\n请问文中提到的DeepSeek-V3的MLA机制有什么特点？"
        
        return full_text[:total_chars] + question, "MLA机制将KV缓存压缩了32倍"
    
    @staticmethod
    def create_multi_hop_reasoning(context_length: int) -> Tuple[str, str]:
        """Create multi-hop reasoning test"""
        facts = [
            "张三是一位软件工程师，他在深度学习公司工作。",
            "深度学习公司专注于开发大语言模型技术。",
            "大语言模型技术中，RetroInfer是一项重要的优化技术。",
            "RetroInfer技术由李四博士发明，可以大幅提升推理效率。",
            "李四博士曾经是张三的导师。"
        ]
        
        # Repeat facts to reach target length
        context = "\n".join(facts * (context_length // 200))
        question = "\n\n请问张三的导师发明了什么技术？这项技术有什么作用？"
        
        return context + question, "李四博士发明了RetroInfer技术，可以大幅提升推理效率"
    
    @staticmethod
    def create_summarization_task(context_length: int) -> Tuple[str, str]:
        """Create summarization test"""
        article = """
        DeepSeek-V3代表了大语言模型架构的重大创新。通过引入Multi-head Latent Attention (MLA)机制，
        模型实现了极致的内存优化。MLA的核心思想是将Key和Value向量投影到低维潜在空间，
        实现了32倍的压缩比，将KV缓存从原本需要的大量内存压缩到仅占原来的3%。
        
        同时，DeepSeek-V3采用了细粒度的MoE架构，拥有671B总参数但每次前向传播仅激活37B参数，
        激活率仅为5.5%。这种设计使得模型在保持强大能力的同时，大幅降低了计算成本。
        
        RetroInfer的集成进一步提升了模型的长序列处理能力。通过将KV缓存视为向量数据库，
        并使用高效的检索算法，模型可以处理超过100K tokens的超长上下文。
        """
        
        # Repeat to reach target length
        full_article = article * (context_length // 500)
        question = "\n\n请用三句话总结上述内容的核心要点。"
        
        return full_article + question, "DeepSeek-V3通过MLA机制实现32倍KV缓存压缩；采用MoE架构仅激活5.5%参数；集成RetroInfer支持超长上下文处理。"


class QualityEvaluator:
    """Evaluate generation quality"""
    
    @staticmethod
    def evaluate_factual_accuracy(generated: str, expected: str) -> float:
        """Simple factual accuracy check"""
        # Extract key terms from expected answer
        key_terms = expected.split()
        found_terms = sum(1 for term in key_terms if term in generated)
        return found_terms / len(key_terms) if key_terms else 0.0
    
    @staticmethod
    def evaluate_fluency(text: str) -> float:
        """Evaluate text fluency (simplified)"""
        # Check for basic fluency indicators
        sentences = text.split('。')
        if len(sentences) < 2:
            return 0.5
        
        # Check average sentence length
        avg_length = np.mean([len(s) for s in sentences if s])
        if 10 <= avg_length <= 50:
            return 1.0
        elif 5 <= avg_length <= 100:
            return 0.8
        else:
            return 0.6
    
    @staticmethod
    def evaluate_coherence(text: str) -> float:
        """Evaluate text coherence (simplified)"""
        # Check for logical connectors
        connectors = ['因此', '所以', '但是', '然而', '同时', '另外', '首先', '其次', '最后']
        connector_count = sum(1 for conn in connectors if conn in text)
        
        # More connectors indicate better structure
        return min(1.0, connector_count * 0.2 + 0.4)


class DeepSeekRetroInferTester:
    """Main test orchestrator"""
    
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-V3", model_version: str = "v3"):
        self.model_name = model_name
        self.model_version = model_version
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.memory_profiler = MemoryProfiler()
        self.data_generator = TestDataGenerator()
        self.quality_evaluator = QualityEvaluator()
        self.results = []
        
    def generate_retroinfer_config(self, context_length: int) -> Dict:
        """Generate optimal RetroInfer configuration based on context length"""
        # Load base config
        config_path = os.path.join(PROJECT_ROOT, "config/DeepSeek-V3.json")
        with open(config_path, "r") as f:
            base_config = json.load(f)
        
        # Get version-specific config
        if self.model_version == "v3":
            version_key = "DeepSeek-V3"
        elif self.model_version == "v2":
            version_key = "DeepSeek-V2"
        elif self.model_version == "v2-lite":
            version_key = "DeepSeek-V2-Lite"
        else:
            version_key = "DeepSeek-V3"  # Default
        
        # Calculate optimal parameters
        n_clusters = max(int(context_length / 16), 32)
        n_segments = max(int(context_length / 8192), 1)
        
        # Align to segment boundaries
        lower = (n_clusters // (n_segments * 32)) * (n_segments * 32)
        upper = lower + (n_segments * 32)
        n_clusters = lower if abs(n_clusters - lower) <= abs(n_clusters - upper) else upper
        
        # Dynamic nprobe based on clusters
        nprobe = max(int(n_clusters * 0.018), 8)
        
        # Build config
        retroinfer_config = base_config[version_key]["RetroInfer"].copy()
        retroinfer_config.update({
            "n_centroids": n_clusters,
            "n_segment": n_segments,
            "nprobe": nprobe,
            "cache_cluster_num": min(nprobe * 3, 64),
            "max_compute_cluster_num": max(int(n_clusters / 4), nprobe)
        })
        
        return {
            "RetroInfer": retroinfer_config,
            "MLA": base_config[version_key]["MLA"]
        }
    
    def test_model(
        self,
        attention_type: str,
        test_data: Tuple[str, str],
        max_new_tokens: int = 100,
        model_version: str = None
    ) -> TestResult:
        """Test model with given configuration"""
        if model_version is None:
            model_version = self.model_version
            
        input_text, expected = test_data
        
        print(colored(f"\n=== Testing {attention_type} with {model_version} ===", "cyan", attrs=["bold"]))
        
        # Initialize model
        model = DeepSeekModel(
            model_name=self.model_name,
            max_length=150000,
            dtype=torch.float16,
            device_map="auto",
            model_version=model_version
        )
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=120000)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        print(f"Input length: {input_ids.shape[1]} tokens")
        
        # Generate config
        if attention_type == "RetroInfer":
            attn_config = self.generate_retroinfer_config(input_ids.shape[1])
            print(f"RetroInfer config: n_centroids={attn_config['RetroInfer']['n_centroids']}, "
                  f"nprobe={attn_config['RetroInfer']['nprobe']}")
        else:
            attn_config = {"Full_Flash_Attn": {}}
        
        # Profile memory and performance
        def inference():
            start_time = time.time()
            first_token_time = None
            
            # Custom generate for measuring first token latency
            outputs = model.generate(
                attention_type=attention_type,
                inputs_ids=input_ids.to(model.layers[0].device),
                attention_masks=attention_mask.to(model.layers[0].device),
                max_new_length=max_new_tokens,
                attn_config=attn_config
            )
            
            end_time = time.time()
            return outputs, end_time - start_time
        
        (outputs, inference_time), memory_stats = self.memory_profiler.profile_memory_during_inference(inference)
        
        # Decode output
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Evaluate quality
        quality_scores = {
            "factual_accuracy": self.quality_evaluator.evaluate_factual_accuracy(generated_text, expected),
            "fluency": self.quality_evaluator.evaluate_fluency(generated_text),
            "coherence": self.quality_evaluator.evaluate_coherence(generated_text)
        }
        
        # Create result
        result = TestResult(
            method=attention_type,
            model_version=model_version,
            input_tokens=input_ids.shape[1],
            output_tokens=max_new_tokens,
            inference_time=inference_time,
            tokens_per_second=max_new_tokens / inference_time,
            gpu_memory_used=memory_stats["gpu_delta"],
            cpu_memory_used=memory_stats["cpu_delta"],
            peak_gpu_memory=memory_stats["peak_gpu"],
            first_token_latency=0.0,  # TODO: Implement proper measurement
            generated_text=generated_text[-1000:],  # Keep last 1000 chars
            retroinfer_params=attn_config.get("RetroInfer") if attention_type == "RetroInfer" else None,
            quality_scores=quality_scores
        )
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return result
    
    def run_benchmark_suite(
        self,
        test_scenarios: List[str] = ["needle", "reasoning", "summarization"],
        context_lengths: List[int] = [10000, 50000, 100000],
        compare_methods: bool = True
    ):
        """Run comprehensive benchmark suite"""
        print(colored("\n🚀 Starting DeepSeek-V3 + RetroInfer Benchmark Suite", "magenta", attrs=["bold"]))
        print("=" * 80)
        
        # Test matrix
        for scenario in test_scenarios:
            for context_length in context_lengths:
                print(colored(f"\n📊 Testing {scenario} with {context_length} tokens", "yellow"))
                
                # Generate test data
                if scenario == "needle":
                    test_data = self.data_generator.create_needle_in_haystack(context_length)
                elif scenario == "reasoning":
                    test_data = self.data_generator.create_multi_hop_reasoning(context_length)
                elif scenario == "summarization":
                    test_data = self.data_generator.create_summarization_task(context_length)
                
                # Test RetroInfer
                result_retro = self.test_model("RetroInfer", test_data)
                self.results.append(result_retro)
                
                # Optionally test standard attention
                if compare_methods and context_length <= 50000:  # Limit to avoid OOM
                    time.sleep(5)  # Allow memory cleanup
                    result_standard = self.test_model("Full_Flash_Attn", test_data)
                    self.results.append(result_standard)
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print(colored("\n📈 Generating Comprehensive Report", "green", attrs=["bold"]))
        print("=" * 80)
        
        # Create report directory
        report_dir = os.path.join(PROJECT_ROOT, "deepseek_retroinfer_report")
        os.makedirs(report_dir, exist_ok=True)
        
        # 1. Performance Summary
        self._generate_performance_summary()
        
        # 2. Memory Usage Analysis
        self._plot_memory_usage()
        
        # 3. Quality Metrics
        self._analyze_quality_metrics()
        
        # 4. Detailed Results JSON
        self._save_detailed_results(report_dir)
        
        print(colored(f"\n✅ Report saved to: {report_dir}", "green"))
    
    def _generate_performance_summary(self):
        """Generate performance summary table"""
        print("\n📊 Performance Summary")
        print("-" * 80)
        
        # Group results by method
        method_stats = defaultdict(list)
        for result in self.results:
            method_stats[result.method].append(result)
        
        # Calculate averages
        for method, results in method_stats.items():
            avg_speed = np.mean([r.tokens_per_second for r in results])
            avg_gpu_mem = np.mean([r.gpu_memory_used for r in results])
            avg_quality = np.mean([np.mean(list(r.quality_scores.values())) for r in results])
            
            print(f"\n{method}:")
            print(f"  • Average Speed: {avg_speed:.2f} tokens/sec")
            print(f"  • Average GPU Memory: {avg_gpu_mem:.2f} GB")
            print(f"  • Average Quality Score: {avg_quality:.2%}")
            
            if method == "RetroInfer":
                # Show RetroInfer specific stats
                cluster_sizes = [r.retroinfer_params['n_centroids'] for r in results if r.retroinfer_params]
                print(f"  • Cluster Sizes Used: {set(cluster_sizes)}")
    
    def _plot_memory_usage(self):
        """Plot memory usage comparison"""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # GPU Memory by context length
            retro_results = [r for r in self.results if r.method == "RetroInfer"]
            standard_results = [r for r in self.results if r.method == "Full_Flash_Attn"]
            
            if retro_results:
                contexts = [r.input_tokens for r in retro_results]
                gpu_mems = [r.gpu_memory_used for r in retro_results]
                ax1.scatter(contexts, gpu_mems, label="RetroInfer", color="blue", s=100)
            
            if standard_results:
                contexts = [r.input_tokens for r in standard_results]
                gpu_mems = [r.gpu_memory_used for r in standard_results]
                ax1.scatter(contexts, gpu_mems, label="Standard", color="red", s=100)
            
            ax1.set_xlabel("Context Length (tokens)")
            ax1.set_ylabel("GPU Memory Used (GB)")
            ax1.set_title("GPU Memory Usage vs Context Length")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Speed comparison
            if retro_results:
                contexts = [r.input_tokens for r in retro_results]
                speeds = [r.tokens_per_second for r in retro_results]
                ax2.scatter(contexts, speeds, label="RetroInfer", color="blue", s=100)
            
            if standard_results:
                contexts = [r.input_tokens for r in standard_results]
                speeds = [r.tokens_per_second for r in standard_results]
                ax2.scatter(contexts, speeds, label="Standard", color="red", s=100)
            
            ax2.set_xlabel("Context Length (tokens)")
            ax2.set_ylabel("Generation Speed (tokens/sec)")
            ax2.set_title("Generation Speed vs Context Length")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            report_dir = os.path.join(PROJECT_ROOT, "deepseek_retroinfer_report")
            plt.savefig(os.path.join(report_dir, "performance_comparison.png"), dpi=150)
            plt.close()
            
        except ImportError:
            print("Warning: matplotlib not available for plotting")
    
    def _analyze_quality_metrics(self):
        """Analyze generation quality"""
        print("\n📝 Quality Analysis")
        print("-" * 80)
        
        # Compare quality between methods
        method_quality = defaultdict(lambda: defaultdict(list))
        
        for result in self.results:
            if result.quality_scores:
                for metric, score in result.quality_scores.items():
                    method_quality[result.method][metric].append(score)
        
        # Print comparison
        for method, metrics in method_quality.items():
            print(f"\n{method} Quality Scores:")
            for metric, scores in metrics.items():
                avg_score = np.mean(scores)
                print(f"  • {metric}: {avg_score:.2%} (±{np.std(scores):.2%})")
    
    def _save_detailed_results(self, report_dir: str):
        """Save detailed results to JSON"""
        results_data = []
        for result in self.results:
            results_data.append({
                "method": result.method,
                "model_version": result.model_version,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "inference_time": result.inference_time,
                "tokens_per_second": result.tokens_per_second,
                "gpu_memory_used": result.gpu_memory_used,
                "cpu_memory_used": result.cpu_memory_used,
                "peak_gpu_memory": result.peak_gpu_memory,
                "quality_scores": result.quality_scores,
                "retroinfer_params": result.retroinfer_params,
                "timestamp": datetime.now().isoformat()
            })
        
        with open(os.path.join(report_dir, "detailed_results.json"), "w") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive DeepSeek-V3 + RetroInfer Testing Suite"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="deepseek-ai/DeepSeek-V3",
        help="Model name or path"
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v3",
        choices=["v2", "v2-lite", "v3"],
        help="Model version to use"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["needle", "reasoning", "summarization"],
        choices=["needle", "reasoning", "summarization"],
        help="Test scenarios to run"
    )
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=[10000, 50000],
        help="Context lengths to test"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Number of tokens to generate"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with standard attention (memory intensive)"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with minimal scenarios"
    )
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick_test:
        args.scenarios = ["needle"]
        args.context_lengths = [10000]
        args.compare = False
    
    # Initialize tester
    tester = DeepSeekRetroInferTester(args.model, args.model_version)
    
    # Run benchmark suite
    tester.run_benchmark_suite(
        test_scenarios=args.scenarios,
        context_lengths=args.context_lengths,
        compare_methods=args.compare
    )
    
    print(colored("\n✨ All tests completed successfully!", "green", attrs=["bold"]))


if __name__ == "__main__":
    main()