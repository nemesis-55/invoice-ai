#!/usr/bin/env python3
"""
Configuration Validator for MiniCPM-V-4.5 Upgrade

This script validates that all configuration files are correctly set up
for training with MiniCPM-V-4.5 and LLamaFactory.
"""

import json
import os
import sys
from pathlib import Path
import yaml


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_model_references():
    """Check if all files reference the correct model."""
    print_section("Checking Model References")
    
    issues = []
    correct_model = "openbmb/MiniCPM-V-4_5"
    correct_llm_type = "qwen3"
    
    # Check finetune.py
    finetune_path = "training/finetune.py"
    if os.path.exists(finetune_path):
        with open(finetune_path, 'r') as f:
            content = f.read()
            if correct_model in content:
                print(f"✓ {finetune_path}: Model reference correct")
            else:
                issues.append(f"✗ {finetune_path}: Model reference may be incorrect")
                
            if f'llm_type: str = field(default="{correct_llm_type}")' in content:
                print(f"✓ {finetune_path}: LLM type correct")
            else:
                issues.append(f"✗ {finetune_path}: LLM type may be incorrect")
    
    # Check shell scripts
    for script in ["training/finetune_ds.sh", "training/finetune_lora.sh"]:
        if os.path.exists(script):
            with open(script, 'r') as f:
                content = f.read()
                if correct_model in content:
                    print(f"✓ {script}: Model reference correct")
                else:
                    print(f"⚠ {script}: Model not found (may use custom fine-tuned model)")
                    
                if f'LLM_TYPE="{correct_llm_type}"' in content:
                    print(f"✓ {script}: LLM type correct")
                else:
                    issues.append(f"✗ {script}: LLM type may be incorrect")
    
    return issues


def check_llamafactory_configs():
    """Check LLamaFactory configuration files."""
    print_section("Checking LLamaFactory Configurations")
    
    issues = []
    config_dir = Path("llamafactory_configs")
    
    if not config_dir.exists():
        issues.append("✗ llamafactory_configs directory not found")
        return issues
    
    # Check dataset_info.json
    dataset_info_path = config_dir / "dataset_info.json"
    if dataset_info_path.exists():
        try:
            with open(dataset_info_path, 'r') as f:
                dataset_info = json.load(f)
                if "invoice_training" in dataset_info:
                    print("✓ dataset_info.json: invoice_training dataset registered")
                else:
                    issues.append("✗ dataset_info.json: invoice_training dataset not found")
        except json.JSONDecodeError as e:
            issues.append(f"✗ dataset_info.json: Invalid JSON - {e}")
    else:
        issues.append("✗ dataset_info.json not found")
    
    # Check YAML configs
    for config_file in ["minicpm_v45_lora.yaml", "minicpm_v45_full.yaml"]:
        config_path = config_dir / config_file
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                    # Check model reference
                    if config.get("model_name_or_path") == "openbmb/MiniCPM-V-4_5":
                        print(f"✓ {config_file}: Model reference correct")
                    else:
                        issues.append(f"✗ {config_file}: Model reference incorrect")
                    
                    # Check dataset reference
                    if config.get("dataset") == "invoice_training":
                        print(f"✓ {config_file}: Dataset reference correct")
                    else:
                        issues.append(f"✗ {config_file}: Dataset reference missing")
                        
            except yaml.YAMLError as e:
                issues.append(f"✗ {config_file}: Invalid YAML - {e}")
        else:
            issues.append(f"✗ {config_file} not found")
    
    return issues


def check_training_scripts():
    """Check training scripts exist and are executable."""
    print_section("Checking Training Scripts")
    
    issues = []
    
    scripts = [
        "training/finetune_ds.sh",
        "training/finetune_lora.sh",
        "training/train_llamafactory_lora.sh",
        "training/train_llamafactory_full.sh"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            if os.access(script, os.X_OK):
                print(f"✓ {script}: Exists and executable")
            else:
                print(f"⚠ {script}: Exists but not executable (run: chmod +x {script})")
        else:
            issues.append(f"✗ {script}: Not found")
    
    return issues


def check_dataset_format():
    """Check if sample dataset follows the correct format."""
    print_section("Checking Dataset Format")
    
    issues = []
    
    # Check if sample data files exist
    data_files = ["data/train_data.json", "data/test_data.json"]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    
                    if not isinstance(data, list):
                        issues.append(f"✗ {data_file}: Should be a list")
                        continue
                    
                    if len(data) == 0:
                        print(f"⚠ {data_file}: Empty dataset")
                        continue
                    
                    # Check first sample
                    sample = data[0]
                    required_keys = ["id", "image", "conversations"]
                    
                    missing_keys = [key for key in required_keys if key not in sample]
                    if missing_keys:
                        issues.append(f"✗ {data_file}: Missing keys: {missing_keys}")
                    else:
                        print(f"✓ {data_file}: Format correct ({len(data)} samples)")
                        
                        # Check conversations format
                        convs = sample.get("conversations", [])
                        if len(convs) >= 2:
                            if convs[0].get("role") == "user" and convs[1].get("role") == "assistant":
                                print(f"  ✓ Conversation format valid")
                            else:
                                issues.append(f"✗ {data_file}: Invalid conversation roles")
                        
            except json.JSONDecodeError as e:
                issues.append(f"✗ {data_file}: Invalid JSON - {e}")
        else:
            print(f"⚠ {data_file}: Not found (will be created during data preparation)")
    
    return issues


def check_documentation():
    """Check if documentation files exist."""
    print_section("Checking Documentation")
    
    issues = []
    
    docs = {
        "UPGRADE_NOTES.md": "Upgrade documentation",
        "LLAMAFACTORY_GUIDE.md": "LLamaFactory guide",
        "README.md": "Main README"
    }
    
    for doc, description in docs.items():
        if os.path.exists(doc):
            print(f"✓ {doc}: {description} exists")
        else:
            issues.append(f"✗ {doc}: Not found")
    
    return issues


def check_dependencies():
    """Check if requirements.txt is updated."""
    print_section("Checking Dependencies")
    
    issues = []
    
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", 'r') as f:
            content = f.read()
            
            required_packages = [
                "transformers",
                "torch",
                "peft",
                "deepspeed",
                "accelerate"
            ]
            
            for package in required_packages:
                if package in content:
                    print(f"✓ {package} listed in requirements.txt")
                else:
                    issues.append(f"✗ {package} not found in requirements.txt")
            
            if "llamafactory" in content.lower():
                print("✓ LLamaFactory mentioned in requirements.txt")
            else:
                print("⚠ LLamaFactory not in requirements.txt (may be installed separately)")
    else:
        issues.append("✗ requirements.txt not found")
    
    return issues


def main():
    """Run all validation checks."""
    print("\n" + "=" * 60)
    print("  MiniCPM-V-4.5 Configuration Validator")
    print("=" * 60)
    
    all_issues = []
    
    # Run all checks
    all_issues.extend(check_model_references())
    all_issues.extend(check_llamafactory_configs())
    all_issues.extend(check_training_scripts())
    all_issues.extend(check_dataset_format())
    all_issues.extend(check_documentation())
    all_issues.extend(check_dependencies())
    
    # Print summary
    print_section("Validation Summary")
    
    if all_issues:
        print(f"\n❌ Found {len(all_issues)} issue(s):\n")
        for issue in all_issues:
            print(f"  {issue}")
        print("\nPlease fix these issues before proceeding with training.")
        return 1
    else:
        print("\n✅ All checks passed!")
        print("\nYour configuration is ready for training with MiniCPM-V-4.5.")
        print("\nNext steps:")
        print("  1. Prepare your dataset: python prepare_data/create_training_data.py")
        print("  2. Choose training method:")
        print("     - Native LoRA: bash training/finetune_lora.sh")
        print("     - Native Full: bash training/finetune_ds.sh")
        print("     - LLamaFactory LoRA: bash training/train_llamafactory_lora.sh")
        print("     - LLamaFactory Full: bash training/train_llamafactory_full.sh")
        print("     - Web UI: llamafactory-cli webui")
        return 0


if __name__ == "__main__":
    sys.exit(main())
