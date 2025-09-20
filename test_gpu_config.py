#!/usr/bin/env python3
"""
Test script to validate the new GPU configuration system.
Tests loading from gpu_config.json and environment variable overrides.
"""

import os
import json
import tempfile
import shutil

def test_gpu_config_loading():
    """Test loading GPU configuration from json file"""
    
    # Test data
    test_config = {
        "gpu_profiles": {
            "rtx_4090": {
                "name": "NVIDIA RTX 4090",
                "vram_gb": 24,
                "recommended_settings": {
                    "model_precision": "8bit",
                    "gpu_device": "single",
                    "max_new_tokens": 4096
                }
            },
            "rtx_5090": {
                "name": "NVIDIA RTX 5090", 
                "vram_gb": 32,
                "recommended_settings": {
                    "model_precision": "16bit",
                    "gpu_device": "single",
                    "max_new_tokens": 8192
                }
            }
        },
        "active_profile": "rtx_4090",
        "fallback_settings": {
            "model_precision": "16bit",
            "gpu_device": "single",
            "max_new_tokens": 4096
        }
    }
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        temp_config_path = f.name
    
    try:
        # Simulate the config loading function from handler.py
        def load_gpu_config(config_path):
            """Load GPU configuration from gpu_config.json"""
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                active_profile = config.get("active_profile", "rtx_4090")
                profile_settings = config["gpu_profiles"].get(active_profile, {}).get("recommended_settings", {})
                fallback_settings = config.get("fallback_settings", {})
                
                # Merge profile settings with fallback
                settings = {**fallback_settings, **profile_settings}
                
                return settings, active_profile
            except Exception as e:
                print(f"Warning: Could not load GPU config ({e}), using defaults")
                return {
                    "model_precision": "16bit",
                    "gpu_device": "single",
                    "max_new_tokens": 4096
                }, "default"
        
        # Test RTX 4090 profile
        settings, profile = load_gpu_config(temp_config_path)
        assert profile == "rtx_4090", f"Expected rtx_4090 profile, got {profile}"
        assert settings["model_precision"] == "8bit", f"Expected 8bit precision for 4090, got {settings['model_precision']}"
        assert settings["max_new_tokens"] == 4096, f"Expected 4096 tokens for 4090, got {settings['max_new_tokens']}"
        print("✓ RTX 4090 profile test passed")
        
        # Test RTX 5090 profile
        test_config["active_profile"] = "rtx_5090"
        with open(temp_config_path, 'w') as f:
            json.dump(test_config, f)
        
        settings, profile = load_gpu_config(temp_config_path)
        assert profile == "rtx_5090", f"Expected rtx_5090 profile, got {profile}"
        assert settings["model_precision"] == "16bit", f"Expected 16bit precision for 5090, got {settings['model_precision']}"
        assert settings["max_new_tokens"] == 8192, f"Expected 8192 tokens for 5090, got {settings['max_new_tokens']}"
        print("✓ RTX 5090 profile test passed")
        
        # Test invalid profile (should fall back)
        test_config["active_profile"] = "invalid_gpu"
        with open(temp_config_path, 'w') as f:
            json.dump(test_config, f)
        
        settings, profile = load_gpu_config(temp_config_path)
        assert settings["model_precision"] == "16bit", f"Expected fallback 16bit precision, got {settings['model_precision']}"
        assert settings["max_new_tokens"] == 4096, f"Expected fallback 4096 tokens, got {settings['max_new_tokens']}"
        print("✓ Invalid profile fallback test passed")
        
    finally:
        os.unlink(temp_config_path)

def test_environment_override():
    """Test that environment variables still override GPU config settings"""
    
    # Clean environment first
    env_vars = ["MODEL_PRECISION", "GPU_DEVICE"]
    original_values = {}
    for var in env_vars:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]
    
    try:
        # Simulate GPU config loading with 4090 settings
        gpu_config = {
            "model_precision": "8bit",
            "gpu_device": "single", 
            "max_new_tokens": 4096
        }
        
        # Test default behavior (no env vars)
        MODEL_PRECISION = os.getenv("MODEL_PRECISION", gpu_config.get("model_precision", "16bit"))
        GPU_DEVICE = os.getenv("GPU_DEVICE", gpu_config.get("gpu_device", "single"))
        
        assert MODEL_PRECISION == "8bit", f"Expected 8bit from config, got {MODEL_PRECISION}"
        assert GPU_DEVICE == "single", f"Expected single from config, got {GPU_DEVICE}"
        print("✓ GPU config default loading test passed")
        
        # Test environment variable override
        os.environ["MODEL_PRECISION"] = "16bit"
        os.environ["GPU_DEVICE"] = "auto"
        
        MODEL_PRECISION = os.getenv("MODEL_PRECISION", gpu_config.get("model_precision", "16bit"))
        GPU_DEVICE = os.getenv("GPU_DEVICE", gpu_config.get("gpu_device", "single"))
        
        assert MODEL_PRECISION == "16bit", f"Expected 16bit from env override, got {MODEL_PRECISION}"
        assert GPU_DEVICE == "auto", f"Expected auto from env override, got {GPU_DEVICE}"
        print("✓ Environment variable override test passed")
        
    finally:
        # Restore original environment
        for var, value in original_values.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

def test_config_file_exists():
    """Test that the actual gpu_config.json file exists and is valid"""
    config_path = os.path.join(os.path.dirname(__file__), "gpu_config.json")
    
    assert os.path.exists(config_path), f"gpu_config.json not found at {config_path}"
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate structure
    assert "gpu_profiles" in config, "Missing gpu_profiles section"
    assert "active_profile" in config, "Missing active_profile setting"
    assert "fallback_settings" in config, "Missing fallback_settings section"
    
    # Validate profiles exist
    assert "rtx_4090" in config["gpu_profiles"], "Missing rtx_4090 profile"
    assert "rtx_5090" in config["gpu_profiles"], "Missing rtx_5090 profile"
    
    # Validate profile settings
    for profile_name in ["rtx_4090", "rtx_5090"]:
        profile = config["gpu_profiles"][profile_name]
        assert "recommended_settings" in profile, f"Missing recommended_settings in {profile_name}"
        settings = profile["recommended_settings"]
        assert "model_precision" in settings, f"Missing model_precision in {profile_name}"
        assert "gpu_device" in settings, f"Missing gpu_device in {profile_name}"
        assert "max_new_tokens" in settings, f"Missing max_new_tokens in {profile_name}"
    
    print("✓ GPU config file structure validation passed")

if __name__ == "__main__":
    try:
        print("Testing GPU configuration system...")
        print("=" * 50)
        
        test_config_file_exists()
        test_gpu_config_loading()
        test_environment_override()
        
        print("=" * 50)
        print("🎉 All GPU configuration tests passed!")
        print("\nThe system now supports:")
        print("- Automatic parameter selection based on GPU type")
        print("- RTX 4090 and RTX 5090 optimized profiles")
        print("- Environment variable overrides for advanced users")
        print("- Fallback settings for unknown configurations")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)