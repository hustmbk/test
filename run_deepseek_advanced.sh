#!/bin/bash
# Enhanced DeepSeek-V3 + RetroInfer Test Runner
# This script provides a comprehensive testing interface with multiple options

set -e  # Exit on error

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Print banner
print_banner() {
    echo -e "${MAGENTA}${BOLD}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║         DeepSeek-V3 + RetroInfer Test Suite v2.0             ║"
    echo "║                                                              ║"
    echo "║  Advanced testing framework for MLA + RetroInfer integration ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check system requirements
check_requirements() {
    echo -e "${CYAN}Checking system requirements...${NC}"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 is not installed${NC}"
        exit 1
    fi
    
    # Check CUDA
    python3 -c "
import torch
cuda_available = torch.cuda.is_available()
cuda_count = torch.cuda.device_count() if cuda_available else 0
print(f'✓ CUDA Available: {cuda_available}')
print(f'✓ GPU Count: {cuda_count}')
if cuda_count > 0:
    for i in range(cuda_count):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)')
"
    
    # Check available memory
    echo -e "\n${CYAN}System Memory:${NC}"
    free -h | grep -E "^Mem:" | awk '{print "  RAM: " $2 " total, " $7 " available"}'
    
    # Check required packages
    echo -e "\n${CYAN}Checking Python packages...${NC}"
    python3 -c "
import importlib
packages = ['torch', 'transformers', 'numpy', 'psutil', 'termcolor']
missing = []
for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f'  ✓ {pkg}')
    except ImportError:
        print(f'  ❌ {pkg} (missing)')
        missing.append(pkg)
if missing:
    print(f'\\nPlease install missing packages: pip install {\" \".join(missing)}')
    exit(1)
"
}

# Function to run quick test
run_quick_test() {
    echo -e "${GREEN}Running quick test (10K tokens)...${NC}"
    python3 test_deepseek_comprehensive.py \
        --quick-test \
        --max-new-tokens 50
}

# Function to run standard benchmark
run_standard_benchmark() {
    echo -e "${GREEN}Running standard benchmark...${NC}"
    python3 test_deepseek_comprehensive.py \
        --scenarios needle reasoning \
        --context-lengths 10000 50000 \
        --max-new-tokens 100
}

# Function to run full benchmark
run_full_benchmark() {
    echo -e "${YELLOW}⚠️  Warning: Full benchmark requires significant time and resources!${NC}"
    read -p "Continue? (y/n): " confirm
    if [[ $confirm != "y" ]]; then
        return
    fi
    
    python3 test_deepseek_comprehensive.py \
        --scenarios needle reasoning summarization \
        --context-lengths 10000 50000 100000 \
        --max-new-tokens 100 \
        --compare
}

# Function to run custom test
run_custom_test() {
    echo -e "${CYAN}Custom Test Configuration${NC}"
    
    # Select scenarios
    echo -e "\nAvailable scenarios:"
    echo "  1) Needle in Haystack"
    echo "  2) Multi-hop Reasoning"
    echo "  3) Summarization"
    echo "  4) All scenarios"
    read -p "Select scenarios (1-4, comma separated): " scenario_choice
    
    scenarios=""
    if [[ $scenario_choice == *"1"* ]]; then scenarios="$scenarios needle"; fi
    if [[ $scenario_choice == *"2"* ]]; then scenarios="$scenarios reasoning"; fi
    if [[ $scenario_choice == *"3"* ]]; then scenarios="$scenarios summarization"; fi
    if [[ $scenario_choice == *"4"* ]]; then scenarios="needle reasoning summarization"; fi
    
    # Select context lengths
    echo -e "\nEnter context lengths (space separated, e.g., '10000 50000'):"
    read -p "> " context_lengths
    
    # Select max new tokens
    read -p "Max new tokens to generate (default 100): " max_tokens
    max_tokens=${max_tokens:-100}
    
    # Compare with standard attention?
    read -p "Compare with standard attention? (y/n): " compare
    compare_flag=""
    if [[ $compare == "y" ]]; then compare_flag="--compare"; fi
    
    # Run test
    echo -e "\n${GREEN}Running custom test...${NC}"
    python3 test_deepseek_comprehensive.py \
        --scenarios $scenarios \
        --context-lengths $context_lengths \
        --max-new-tokens $max_tokens \
        $compare_flag
}

# Function to run memory profiling
run_memory_profile() {
    echo -e "${CYAN}Memory Profiling Mode${NC}"
    echo "This will profile memory usage during inference..."
    
    read -p "Context length to profile (default 50000): " context_len
    context_len=${context_len:-50000}
    
    python3 -c "
import sys
sys.path.append('.')
from test_deepseek_comprehensive import DeepSeekRetroInferTester, TestDataGenerator

tester = DeepSeekRetroInferTester()
data_gen = TestDataGenerator()
test_data = data_gen.create_needle_in_haystack($context_len)

print('\\nProfiling RetroInfer...')
result = tester.test_model('RetroInfer', test_data, max_new_tokens=50)

print(f'\\nMemory Profile:')
print(f'  GPU Memory Used: {result.gpu_memory_used:.2f} GB')
print(f'  Peak GPU Memory: {result.peak_gpu_memory:.2f} GB')
print(f'  CPU Memory Used: {result.cpu_memory_used:.2f} GB')
print(f'  Generation Speed: {result.tokens_per_second:.2f} tokens/sec')
"
}

# Function to view previous results
view_results() {
    report_dir="deepseek_retroinfer_report"
    if [[ ! -d $report_dir ]]; then
        echo -e "${RED}No previous results found${NC}"
        return
    fi
    
    echo -e "${CYAN}Previous Test Results${NC}"
    
    # Check for results file
    if [[ -f "$report_dir/detailed_results.json" ]]; then
        echo -e "\n${GREEN}Latest results summary:${NC}"
        python3 -c "
import json
with open('$report_dir/detailed_results.json', 'r') as f:
    results = json.load(f)
    
# Group by method
from collections import defaultdict
method_stats = defaultdict(list)
for r in results:
    method_stats[r['method']].append(r)

# Print summary
for method, res_list in method_stats.items():
    avg_speed = sum(r['tokens_per_second'] for r in res_list) / len(res_list)
    avg_gpu = sum(r['gpu_memory_used'] for r in res_list) / len(res_list)
    print(f'\\n{method}:')
    print(f'  Average Speed: {avg_speed:.2f} tokens/sec')
    print(f'  Average GPU Memory: {avg_gpu:.2f} GB')
    print(f'  Tests Run: {len(res_list)}')
"
    fi
    
    # Check for plots
    if [[ -f "$report_dir/performance_comparison.png" ]]; then
        echo -e "\n${GREEN}Performance plots available at:${NC}"
        echo "  $report_dir/performance_comparison.png"
    fi
}

# Function to run interactive mode
run_interactive() {
    echo -e "${CYAN}Interactive Testing Mode${NC}"
    echo "Enter your test prompt (end with Ctrl-D):"
    
    # Create temporary file for input
    temp_file=$(mktemp)
    cat > $temp_file
    
    # Run test with custom input
    python3 -c "
import sys
sys.path.append('.')
from test_deepseek_comprehensive import DeepSeekRetroInferTester
import torch
from transformers import AutoTokenizer

# Read input
with open('$temp_file', 'r') as f:
    input_text = f.read()

# Initialize
tester = DeepSeekRetroInferTester()
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V3')

# Tokenize to check length
tokens = tokenizer.encode(input_text)
print(f'\\nInput length: {len(tokens)} tokens')

# Run test
test_data = (input_text, 'N/A')
result = tester.test_model('RetroInfer', test_data, max_new_tokens=100)

print(f'\\nGeneration completed in {result.inference_time:.2f} seconds')
print(f'Speed: {result.tokens_per_second:.2f} tokens/sec')
print(f'\\nGenerated text:\\n{result.generated_text}')
"
    
    rm -f $temp_file
}

# Main menu
main_menu() {
    while true; do
        echo -e "\n${BOLD}Select Test Mode:${NC}"
        echo "  1) 🚀 Quick Test (10K tokens, fast)"
        echo "  2) 📊 Standard Benchmark (recommended)"
        echo "  3) 🔥 Full Benchmark (all scenarios, comparison)"
        echo "  4) ⚙️  Custom Test Configuration"
        echo "  5) 📈 Memory Profiling"
        echo "  6) 📁 View Previous Results"
        echo "  7) 💬 Interactive Mode"
        echo "  8) ❓ Help"
        echo "  9) 🚪 Exit"
        
        read -p "Enter choice (1-9): " choice
        
        case $choice in
            1) run_quick_test ;;
            2) run_standard_benchmark ;;
            3) run_full_benchmark ;;
            4) run_custom_test ;;
            5) run_memory_profile ;;
            6) view_results ;;
            7) run_interactive ;;
            8) show_help ;;
            9) echo -e "${GREEN}Goodbye!${NC}"; exit 0 ;;
            *) echo -e "${RED}Invalid choice${NC}" ;;
        esac
        
        echo -e "\n${CYAN}Press Enter to continue...${NC}"
        read
    done
}

# Show help
show_help() {
    echo -e "${CYAN}${BOLD}DeepSeek-V3 + RetroInfer Test Suite Help${NC}"
    echo
    echo "This test suite evaluates DeepSeek-V3's performance with RetroInfer optimization."
    echo
    echo "Key Features:"
    echo "  • MLA (Multi-head Latent Attention) - 32x KV cache compression"
    echo "  • MoE (Mixture of Experts) - Only 5.5% parameters activated"
    echo "  • RetroInfer - Vector database approach for long sequences"
    echo
    echo "Test Scenarios:"
    echo "  • Needle in Haystack - Find specific information in long context"
    echo "  • Multi-hop Reasoning - Connect multiple facts"
    echo "  • Summarization - Condense long texts"
    echo
    echo "Performance Metrics:"
    echo "  • Generation speed (tokens/second)"
    echo "  • Memory usage (GPU and CPU)"
    echo "  • Quality scores (accuracy, fluency, coherence)"
    echo
    echo "Tips:"
    echo "  • Start with Quick Test to verify setup"
    echo "  • Standard Benchmark provides good overview"
    echo "  • Full Benchmark requires 80GB+ GPU memory"
}

# Main execution
print_banner
check_requirements
echo
main_menu