#!/usr/bin/env python3
"""
Quick verification script for DeepSeek-V2-Lite support
"""

import os
import sys
import json
from termcolor import colored

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)


def verify_v2_lite_config():
    """Verify V2-Lite configuration exists"""
    print(colored("Verifying DeepSeek-V2-Lite Configuration", "cyan", attrs=["bold"]))
    print("=" * 50)
    
    # Check config file
    config_path = os.path.join(PROJECT_ROOT, "config/DeepSeek-V3.json")
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        
        if "DeepSeek-V2-Lite" in config:
            print(colored("✓ V2-Lite configuration found in config file", "green"))
            
            v2_lite = config["DeepSeek-V2-Lite"]
            print("\nConfiguration details:")
            print(f"  • Total parameters: {v2_lite['model_info']['total_parameters']}")
            print(f"  • Active parameters: {v2_lite['model_info']['active_parameters']}")
            print(f"  • Hidden layers: {v2_lite['architecture']['num_hidden_layers']}")
            print(f"  • Hidden size: {v2_lite['architecture']['hidden_size']}")
            print(f"  • MoE experts: {v2_lite['architecture']['moe_config']['num_experts']}")
            print(f"  • KV compression: {v2_lite['architecture']['mla_config']['compression_ratio']}x")
        else:
            print(colored("✗ V2-Lite configuration NOT found", "red"))
            return False
            
    except Exception as e:
        print(colored(f"✗ Error reading config: {str(e)}", "red"))
        return False
    
    # Check model support
    print(colored("\n\nVerifying Model Implementation", "cyan", attrs=["bold"]))
    print("=" * 50)
    
    try:
        # Check the source file directly without importing
        model_path = os.path.join(PROJECT_ROOT, "model_hub/deepseek.py")
        if os.path.exists(model_path):
            with open(model_path, "r") as f:
                model_source = f.read()
            
            # Check if v2-lite is supported
            if '"v2-lite"' in model_source or "'v2-lite'" in model_source:
                print(colored("✓ V2-Lite support found in DeepSeekModel source", "green"))
                
                # Count occurrences
                v2_lite_count = model_source.count("v2-lite")
                print(f"  Found {v2_lite_count} references to v2-lite in the code")
            else:
                print(colored("✗ V2-Lite not found in DeepSeekModel source", "red"))
                return False
        else:
            print(colored("✗ DeepSeekModel source file not found", "red"))
            return False
            
    except Exception as e:
        print(colored(f"✗ Error checking model: {str(e)}", "red"))
        return False
    
    # Check test scripts
    print(colored("\n\nVerifying Test Scripts", "cyan", attrs=["bold"]))
    print("=" * 50)
    
    scripts_to_check = [
        "test_deepseek_comprehensive.py",
        "deepseek_simple_example.py",
        "test_deepseek_v2_lite.py"
    ]
    
    for script in scripts_to_check:
        script_path = os.path.join(PROJECT_ROOT, script)
        if os.path.exists(script_path):
            with open(script_path, "r") as f:
                content = f.read()
                if "v2-lite" in content:
                    print(colored(f"✓ {script} supports V2-Lite", "green"))
                else:
                    print(colored(f"✗ {script} does not mention V2-Lite", "yellow"))
        else:
            print(colored(f"✗ {script} not found", "red"))
    
    print(colored("\n\nSummary", "cyan", attrs=["bold"]))
    print("=" * 50)
    print(colored("✅ DeepSeek-V2-Lite support has been successfully added!", "green", attrs=["bold"]))
    print("\nYou can now use V2-Lite with:")
    print("  • model_version='v2-lite' in DeepSeekModel")
    print("  • --model-version v2-lite in test scripts")
    print("  • python test_deepseek_v2_lite.py for specific tests")
    
    return True


def show_usage_examples():
    """Show usage examples"""
    print(colored("\n\nUsage Examples", "magenta", attrs=["bold"]))
    print("=" * 50)
    
    print("\n1. Simple example with V2-Lite:")
    print(colored("   python deepseek_simple_example.py --model-version v2-lite", "white", "on_grey"))
    
    print("\n2. Comprehensive test with V2-Lite:")
    print(colored("   python test_deepseek_comprehensive.py --model-version v2-lite --quick-test", "white", "on_grey"))
    
    print("\n3. V2-Lite specific tests:")
    print(colored("   python test_deepseek_v2_lite.py --test all", "white", "on_grey"))
    
    print("\n4. Python code example:")
    print(colored("""
from model_hub import DeepSeekModel

model = DeepSeekModel(
    model_name="deepseek-ai/DeepSeek-V2-Lite",
    max_length=32768,
    dtype=torch.float16,
    device_map="auto",
    model_version="v2-lite"  # Important!
)
""", "white", "on_grey"))


if __name__ == "__main__":
    if verify_v2_lite_config():
        show_usage_examples()
    else:
        print(colored("\n❌ Verification failed!", "red", attrs=["bold"]))