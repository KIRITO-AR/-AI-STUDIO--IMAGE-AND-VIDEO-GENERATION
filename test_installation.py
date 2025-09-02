#!/usr/bin/env python3
"""
Quick installation test for AI Generation Studio.
Run this to verify everything is set up correctly.
"""

import sys
from pathlib import Path

def test_basic_imports():
    """Test if basic Python modules can be imported."""
    print("🔍 Testing basic imports...")
    
    try:
        import logging
        import configparser
        import time
        import threading
        import pathlib
        print("✅ Standard library imports successful")
    except ImportError as e:
        print(f"❌ Standard library import failed: {e}")
        return False
    
    return True

def test_required_packages():
    """Test if required packages are installed."""
    print("📦 Testing required packages...")
    
    packages = [
        ('torch', 'PyTorch'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('streamlit', 'Streamlit'),
    ]
    
    missing = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} missing")
            missing.append(name)
    
    if missing:
        print(f"\n🔧 Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def test_project_structure():
    """Test if project structure is correct."""
    print("📁 Testing project structure...")
    
    required_dirs = [
        'src/core',
        'src/ui', 
        'src/utils',
        'configs',
        'examples',
        'docs',
        'tests'
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)
        else:
            print(f"✅ {dir_path}/ exists")
    
    if missing:
        print(f"\n❌ Missing directories: {', '.join(missing)}")
        return False
    
    return True

def test_core_modules():
    """Test if core modules can be imported."""
    print("🧠 Testing core modules...")
    
    # Add src to path
    sys.path.insert(0, str(Path('src')))
    
    try:
        from utils.config import ConfigManager
        print("✅ Configuration system working")
    except Exception as e:
        print(f"❌ Configuration system failed: {e}")
        return False
    
    try:
        from utils.gpu_utils import GPUDetector
        print("✅ GPU detection system working")
    except Exception as e:
        print(f"❌ GPU detection failed: {e}")
        return False
    
    try:
        from core.model_manager import ModelManager
        print("✅ Model management system working")
    except Exception as e:
        print(f"❌ Model management failed: {e}")
        return False
    
    return True

def test_gpu_detection():
    """Test GPU detection."""
    print("🎮 Testing GPU detection...")
    
    sys.path.insert(0, str(Path('src')))
    
    try:
        from utils.gpu_utils import get_device_info
        device, gpu_detector = get_device_info()
        
        print(f"✅ Device detected: {device}")
        
        if gpu_detector.gpus:
            best_gpu = gpu_detector.get_best_gpu()
            print(f"✅ Best GPU: {best_gpu.name} ({best_gpu.memory_total}MB)")
        else:
            print("ℹ️  No GPU detected - will use CPU")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU detection failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 AI Generation Studio - Installation Test")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_required_packages,
        test_project_structure,
        test_core_modules,
        test_gpu_detection
    ]
    
    results = []
    for test in tests:
        print()
        result = test()
        results.append(result)
        
        if not result:
            print("⚠️  Test failed - check errors above")
    
    print("\n" + "=" * 50)
    
    if all(results):
        print("🎉 All tests passed! Installation successful!")
        print("\n📖 Next steps:")
        print("1. Launch Streamlit: python launch.py --streamlit")
        print("2. Try examples: python examples/simple_generation.py")
        print("3. Open browser to: http://localhost:8501")
    else:
        failed_count = sum(1 for r in results if not r)
        print(f"❌ {failed_count}/{len(tests)} tests failed")
        print("\n🔧 Common fixes:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Check Python version: python --version (need 3.8+)")
        print("- Run setup script: python setup.py")

if __name__ == "__main__":
    main()