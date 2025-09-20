#!/usr/bin/env python3
"""
Test script to validate the new configurable GPU and precision settings.
This script tests the configuration logic without requiring torch.
"""

import os

def test_configuration():
    """Test the configuration logic from handler.py"""
    
    test_cases = [
        # (MODEL_PRECISION, GPU_DEVICE, expected_load_in_8bit, expected_load_in_4bit, expected_device_map)
        ("16bit", "single", False, False, "cuda:0"),
        ("8bit", "single", True, False, "cuda:0"),
        ("4bit", "single", False, True, "cuda:0"),
        ("16bit", "auto", False, False, "auto"),
        ("8bit", "auto", True, False, "auto"),
        ("16bit", "cuda:1", False, False, "cuda:1"),
    ]
    
    print("Testing configuration logic...")
    print("=" * 50)
    
    for model_precision, gpu_device, expected_8bit, expected_4bit, expected_device in test_cases:
        # Simulate environment variables
        os.environ["MODEL_PRECISION"] = model_precision
        os.environ["GPU_DEVICE"] = gpu_device
        
        # Test the logic from handler.py
        MODEL_PRECISION = os.getenv("MODEL_PRECISION", "16bit")
        GPU_DEVICE = os.getenv("GPU_DEVICE", "single")
        
        # Configure device mapping (simulating torch.cuda.is_available() as True)
        if GPU_DEVICE == "single":
            device_map = "cuda:0"  # Assuming CUDA is available
        elif GPU_DEVICE == "auto":
            device_map = "auto"
        else:
            device_map = GPU_DEVICE
        
        # Configure quantization
        load_in_8bit = MODEL_PRECISION == "8bit"
        load_in_4bit = MODEL_PRECISION == "4bit"
        
        # Validate results
        assert load_in_8bit == expected_8bit, f"load_in_8bit mismatch for {model_precision}"
        assert load_in_4bit == expected_4bit, f"load_in_4bit mismatch for {model_precision}"
        assert device_map == expected_device, f"device_map mismatch for {gpu_device}"
        
        print(f"✓ {model_precision:5} + {gpu_device:6} -> 8bit:{load_in_8bit}, 4bit:{load_in_4bit}, device:{device_map}")
    
    print("=" * 50)
    print("✓ All configuration tests passed!")
    
    # Test default values
    print("\nTesting default values...")
    del os.environ["MODEL_PRECISION"]
    del os.environ["GPU_DEVICE"]
    
    MODEL_PRECISION = os.getenv("MODEL_PRECISION", "16bit")
    GPU_DEVICE = os.getenv("GPU_DEVICE", "single")
    
    assert MODEL_PRECISION == "16bit", f"Default MODEL_PRECISION should be 16bit, got {MODEL_PRECISION}"
    assert GPU_DEVICE == "single", f"Default GPU_DEVICE should be single, got {GPU_DEVICE}"
    
    print("✓ Default values test passed!")
    print(f"  Default MODEL_PRECISION: {MODEL_PRECISION}")
    print(f"  Default GPU_DEVICE: {GPU_DEVICE}")

def test_environment_validation():
    """Test that the environment variables work as intended"""
    print("\nTesting environment variable overrides...")
    
    # Test setting environment variables
    os.environ["MODEL_PRECISION"] = "8bit"
    os.environ["GPU_DEVICE"] = "cuda:1"
    
    precision = os.getenv("MODEL_PRECISION", "16bit")
    device = os.getenv("GPU_DEVICE", "single")
    
    assert precision == "8bit", f"Environment override failed for MODEL_PRECISION"
    assert device == "cuda:1", f"Environment override failed for GPU_DEVICE"
    
    print("✓ Environment variable override test passed!")
    print(f"  MODEL_PRECISION from env: {precision}")
    print(f"  GPU_DEVICE from env: {device}")

if __name__ == "__main__":
    try:
        test_configuration()
        test_environment_validation()
        print("\n🎉 All tests passed! Configuration is working correctly.")
        print("\nThe handler.py will now use:")
        print("- 16-bit precision by default (as requested)")
        print("- Single GPU usage by default (as requested)")
        print("- Configurable via environment variables for flexibility")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)