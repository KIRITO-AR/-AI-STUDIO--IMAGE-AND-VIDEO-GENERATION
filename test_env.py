#!/usr/bin/env python3
"""
Simple test to verify the virtual environment is working correctly.
This script will create a small file with the test results.
"""

def test_environment():
    results = []
    
    # Test 1: Basic imports
    try:
        import sys
        results.append(f"Python version: {sys.version}")
    except Exception as e:
        results.append(f"Python test failed: {e}")
    
    # Test 2: PyTorch
    try:
        import torch
        results.append(f"PyTorch: {torch.__version__}")
        results.append(f"CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        results.append(f"PyTorch failed: {e}")
    
    # Test 3: Diffusers
    try:
        import diffusers
        results.append(f"Diffusers: {diffusers.__version__}")
    except Exception as e:
        results.append(f"Diffusers failed: {e}")
    
    # Test 4: Transformers
    try:
        import transformers
        results.append(f"Transformers: {transformers.__version__}")
    except Exception as e:
        results.append(f"Transformers failed: {e}")
    
    # Test 5: Other key packages
    try:
        import PIL
        results.append("PIL: OK")
    except Exception as e:
        results.append(f"PIL failed: {e}")
    
    try:
        import numpy
        results.append("NumPy: OK")
    except Exception as e:
        results.append(f"NumPy failed: {e}")
    
    try:
        import streamlit
        results.append("Streamlit: OK")
    except Exception as e:
        results.append(f"Streamlit failed: {e}")
    
    return results

if __name__ == "__main__":
    results = test_environment()
    
    # Write results to file
    with open("env_test_results.txt", "w") as f:
        f.write("Environment Test Results\n")
        f.write("=" * 30 + "\n")
        for result in results:
            f.write(result + "\n")
    
    print("Environment test completed. Check env_test_results.txt for results.")