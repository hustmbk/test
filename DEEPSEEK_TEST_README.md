# DeepSeek-V3 + RetroInfer Integration Testing

This directory contains comprehensive testing tools for evaluating DeepSeek-V3 models with RetroInfer optimization.

## 🚀 Quick Start

### Simple Example
```bash
python deepseek_simple_example.py
```

### Interactive Testing
```bash
./run_deepseek_advanced.sh
```

## 📁 File Overview

### Core Test Scripts

1. **`test_deepseek_comprehensive.py`** - Main comprehensive test suite
   - Multiple test scenarios (needle-in-haystack, reasoning, summarization)
   - Memory profiling and performance benchmarking
   - Quality evaluation metrics
   - Detailed reporting with visualizations

2. **`deepseek_simple_example.py`** - Basic usage example
   - Simple demonstration of DeepSeek-V3 + RetroInfer
   - Good starting point for understanding the integration

3. **`test_deepseek_retroinfer.py`** - Original test script
   - Comparison between standard attention and RetroInfer
   - Memory usage analysis

### Run Scripts

1. **`run_deepseek_advanced.sh`** - Enhanced interactive test runner
   - Menu-driven interface
   - Multiple test modes (quick, standard, full)
   - Memory profiling options
   - Results visualization

2. **`run_deepseek_test.sh`** - Basic test runner
   - Simple predefined test scenarios

## 🔧 Test Modes

### 1. Quick Test (Recommended for first run)
- Context: 10K tokens
- Fast validation of setup
- Minimal resource usage

### 2. Standard Benchmark
- Scenarios: Needle-in-haystack, Multi-hop reasoning
- Context lengths: 10K, 50K tokens
- Balanced performance evaluation

### 3. Full Benchmark
- All scenarios including summarization
- Context lengths up to 100K tokens
- Comparison with standard attention
- Requires 80GB+ GPU memory

### 4. Custom Test
- Configure your own test parameters
- Choose specific scenarios and context lengths

### 5. Memory Profiling
- Detailed memory usage analysis
- GPU and CPU memory tracking
- Peak memory measurement

## 📊 Test Scenarios

### Needle in Haystack
Tests the model's ability to find specific information in a large context.

### Multi-hop Reasoning
Evaluates the model's capability to connect multiple facts to answer questions.

### Summarization
Assesses the model's ability to condense long texts while preserving key information.

## 🔑 Key Features

### DeepSeek-V3 Innovations
- **MLA (Multi-head Latent Attention)**: 32x KV cache compression
- **MoE (Mixture of Experts)**: Only 5.5% parameters activated (37B/671B)
- **Support for 128K+ context length**

### RetroInfer Optimizations
- Vector database approach for KV cache
- GPU-CPU collaborative computing
- Dynamic clustering based on context length
- Wave-based attention indexing

## 📈 Performance Metrics

The test suite measures:
- **Generation Speed**: Tokens per second
- **Memory Usage**: GPU and CPU memory consumption
- **Quality Scores**: 
  - Factual accuracy
  - Fluency
  - Coherence
- **Latency**: First token and total inference time

## 🛠️ Configuration

### RetroInfer Parameters
The system automatically adjusts parameters based on context length:
- `n_centroids`: Number of clusters (context_length / 16)
- `n_segments`: Number of segments (context_length / 8192)
- `nprobe`: Number of probes (n_centroids * 0.018)

### MLA Configuration
- KV compression ratio: 32x
- GPU cache ratio: 20%
- CPU offload ratio: 80%

## 📊 Output Reports

Test results are saved in `deepseek_retroinfer_report/`:
- `detailed_results.json`: Complete test metrics
- `performance_comparison.png`: Visual performance comparison
- Console output with colored formatting

## 🚨 Requirements

### Hardware
- NVIDIA GPU with 40GB+ memory (recommended)
- 64GB+ system RAM for long contexts
- CUDA 11.0+

### Software
```bash
pip install torch transformers psutil termcolor matplotlib seaborn nvidia-ml-py
```

## 💡 Tips

1. **Start Small**: Begin with quick test to verify setup
2. **Monitor Memory**: Use memory profiling mode for resource planning
3. **Batch Testing**: The comprehensive suite can run multiple scenarios in sequence
4. **Custom Inputs**: Use interactive mode to test specific prompts

## 🐛 Troubleshooting

### Out of Memory
- Reduce context length
- Disable comparison mode
- Use single GPU mode

### Slow Generation
- Check RetroInfer parameters
- Ensure CUDA is properly configured
- Verify model is using GPU

### Import Errors
- Ensure project root is in Python path
- Install all required dependencies
- Check model_hub directory structure

## 📝 Example Usage

### Basic Test
```python
from test_deepseek_comprehensive import DeepSeekRetroInferTester

tester = DeepSeekRetroInferTester("deepseek-ai/DeepSeek-V3")
tester.run_benchmark_suite(
    test_scenarios=["needle"],
    context_lengths=[10000],
    compare_methods=False
)
```

### Custom Scenario
```python
from test_deepseek_comprehensive import TestDataGenerator, DeepSeekRetroInferTester

# Create custom test data
generator = TestDataGenerator()
test_data = generator.create_needle_in_haystack(50000, needle_position=0.7)

# Run test
tester = DeepSeekRetroInferTester()
result = tester.test_model("RetroInfer", test_data, max_new_tokens=100)

print(f"Speed: {result.tokens_per_second:.2f} tokens/sec")
print(f"GPU Memory: {result.gpu_memory_used:.2f} GB")
```

## 🔗 Related Resources

- [DeepSeek Technical Report](https://github.com/deepseek-ai/DeepSeek-V3)
- [RetroInfer Paper](https://arxiv.org/abs/xxxx.xxxxx)
- [MLA Architecture Details](docs/mla_architecture.md)

## 📄 License

This testing suite follows the same license as the RetroInfer project.