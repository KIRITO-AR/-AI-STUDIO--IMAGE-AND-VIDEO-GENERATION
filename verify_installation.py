#!/usr/bin/env python3
"""
Verify that all required packages are properly installed and working.
"""

def main():
    print("Verifying installation...")
    
    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    # Test diffusers
    try:
        import diffusers
        print(f"✅ Diffusers {diffusers.__version__} imported successfully")
    except Exception as e:
        print(f"❌ Diffusers import failed: {e}")
        return False
    
    # Test transformers
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__} imported successfully")
    except Exception as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    # Test other required packages
    try:
        import PIL
        print("✅ PIL (Pillow) imported successfully")
    except Exception as e:
        print(f"❌ PIL import failed: {e}")
        return False
    
    try:
        import numpy
        print("✅ NumPy imported successfully")
    except Exception as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except Exception as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    print("\n🎉 All required packages are properly installed!")
    print("\nYou can now run the AI Generation Studio:")
    print("1. Launch Streamlit: python launch.py --streamlit")
    print("2. Try examples: python examples/simple_generation.py")
    return True

if __name__ == "__main__":
    main()