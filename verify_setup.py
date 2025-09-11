#!/usr/bin/env python3
"""
Verification script for AI Generation Studio setup.
Run this script to verify all required dependencies are properly installed.
"""

import sys
import os

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version}")
    return True

def check_packages():
    """Check if all required packages are installed."""
    required_packages = [
        ("torch", "PyTorch"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("PIL", "Pillow (PIL)"),
        ("numpy", "NumPy"),
        ("streamlit", "Streamlit"),
        ("accelerate", "Accelerate"),
        ("safetensors", "Safetensors"),
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} missing")
            missing_packages.append(name)
    
    return len(missing_packages) == 0

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available with {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("ℹ️  CUDA not available - will use CPU")
        return True
    except ImportError:
        print("ℹ️  PyTorch not available - cannot check CUDA")
        return False

def main():
    """Run all verification checks."""
    print("🔍 AI Generation Studio - Setup Verification")
    print("=" * 50)
    
    # Check Python version
    print("\n1. Checking Python version...")
    python_ok = check_python_version()
    
    # Check packages
    print("\n2. Checking required packages...")
    packages_ok = check_packages()
    
    # Check CUDA
    print("\n3. Checking CUDA support...")
    cuda_ok = check_cuda()
    
    print("\n" + "=" * 50)
    
    if python_ok and packages_ok:
        print("🎉 Setup verification successful!")
        print("\nYou can now run the application:")
        print("- Streamlit UI: python launch.py --streamlit")
        print("- Examples: python examples/simple_generation.py")
        return True
    else:
        print("❌ Setup verification failed!")
        print("\nPlease check the errors above and ensure all dependencies are installed.")
        print("Refer to SOLUTION_SUMMARY.md and FIX_INSTRUCTIONS.md for detailed troubleshooting.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)