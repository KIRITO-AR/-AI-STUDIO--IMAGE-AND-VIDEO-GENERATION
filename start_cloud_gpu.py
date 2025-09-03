#!/usr/bin/env python3
"""
Cloud GPU startup script for optimized Qwen model performance.
Sets proper environment variables and runs diagnostics.
"""

import os
import sys
import logging
from pathlib import Path

def setup_cloud_gpu_environment():
    """Setup optimal environment variables for cloud GPU usage."""
    print("🌟 Setting up Cloud GPU Environment for Qwen...")
    
    # CUDA Memory Management (optimized for cloud GPUs with 64GB)
    cuda_config = {
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,max_split_size_mb:1024,roundup_power2_divisions:16',
        'CUDA_LAUNCH_BLOCKING': '1',  # Prevent timeout issues
        'TORCH_USE_CUDA_DSA': '1',    # Enable device-side assertions for debugging
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',  # Consistent device ordering
        'CUDA_VISIBLE_DEVICES': '0',  # Use first GPU (adjust if needed)
    }
    
    # PyTorch Optimizations
    pytorch_config = {
        'TORCH_CUDNN_BENCHMARK': '1',   # Enable cuDNN benchmark for better performance
        'TORCH_BACKENDS_CUDNN_ENABLED': '1',  # Enable cuDNN
        'OMP_NUM_THREADS': '8',         # Optimize CPU threads
        'MKL_NUM_THREADS': '8',         # Intel MKL optimization
    }
    
    # Disable ONNX to avoid DLL issues on Windows
    onnx_config = {
        'DIFFUSERS_DISABLE_ONNX': '1',
        'DISABLE_ONNX': '1',
    }
    
    # Apply all configurations
    all_configs = {**cuda_config, **pytorch_config, **onnx_config}
    
    for key, value in all_configs.items():
        os.environ[key] = value
        print(f"✅ Set {key}={value}")
    
    print("🎯 Cloud GPU environment configured successfully!")
    return True

def check_python_environment():
    """Check Python and package versions."""
    print("\\n🐍 Checking Python Environment...")
    
    print(f"📍 Python Version: {sys.version}")
    print(f"📍 Python Executable: {sys.executable}")
    
    # Check required packages
    required_packages = [
        'torch',
        'diffusers', 
        'transformers',
        'accelerate',
        'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - installed")
        except ImportError:
            print(f"❌ {package} - missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\\n⚠️  Missing packages: {missing_packages}")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def run_gpu_diagnostics():
    """Run comprehensive GPU diagnostics."""
    print("\\n🔧 Running GPU Diagnostics...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available!")
            return False
        
        device_count = torch.cuda.device_count()
        print(f"🔢 CUDA devices: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3
            
            print(f"\\n🌟 GPU {i}: {device_name}")
            print(f"  💾 Total Memory: {total_memory:.2f}GB")
            print(f"  🔧 Compute Capability: {props.major}.{props.minor}")
            print(f"  🔄 Multiprocessors: {props.multi_processor_count}")
            
            # Test basic CUDA operations
            try:
                # Create a small tensor to test GPU
                test_tensor = torch.randn(100, 100, device=f'cuda:{i}')
                result = torch.matmul(test_tensor, test_tensor.T)
                torch.cuda.synchronize()
                print(f"  ✅ Basic CUDA operations working")
                
                # Check memory
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                free = total_memory - allocated
                
                print(f"  📊 Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Free: {free:.2f}GB")
                
                # Cloud GPU detection
                if total_memory >= 48.0:
                    print(f"  ☁️  CLOUD GPU DETECTED - Optimized for Qwen!")
                elif total_memory >= 24.0:
                    print(f"  🎮 HIGH-END GPU - Good for Qwen")
                else:
                    print(f"  🖥️  STANDARD GPU - May need optimization")
                
            except Exception as gpu_error:
                print(f"  ❌ GPU test failed: {gpu_error}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ GPU diagnostics failed: {e}")
        return False

def test_model_loading():
    """Test basic model manager functionality."""
    print("\\n🎯 Testing Model Manager...")
    
    try:
        # Add src to path
        current_dir = Path(__file__).resolve().parent
        src_dir = current_dir / 'src'
        sys.path.insert(0, str(src_dir))
        
        from core.model_manager import ModelManager
        
        # Initialize model manager
        model_manager = ModelManager()
        print("✅ Model manager initialized")
        
        # Check Qwen compatibility
        qwen_compat = model_manager.check_model_compatibility("qwen_image")
        print(f"🔍 Qwen compatibility: {qwen_compat.get('compatible', False)}")
        
        if qwen_compat.get('compatible', False):
            print("✅ Qwen is compatible with your GPU!")
            print("💡 Recommendations:")
            for rec in qwen_compat.get('recommendations', []):
                print(f"  - {rec}")
        else:
            print(f"⚠️  Qwen compatibility issue: {qwen_compat.get('reason', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model manager test failed: {e}")
        return False

def main():
    """Main startup sequence."""
    print("🚀 Cloud GPU Startup Script for Qwen")
    print("=" * 50)
    
    steps = [
        ("Environment Setup", setup_cloud_gpu_environment),
        ("Python Environment Check", check_python_environment),
        ("GPU Diagnostics", run_gpu_diagnostics),
        ("Model Manager Test", test_model_loading)
    ]
    
    all_passed = True
    
    for step_name, step_func in steps:
        print(f"\\n▶️  {step_name}...")
        try:
            success = step_func()
            if success:
                print(f"✅ {step_name} completed successfully")
            else:
                print(f"❌ {step_name} failed")
                all_passed = False
        except Exception as e:
            print(f"❌ {step_name} crashed: {e}")
            all_passed = False
    
    print("\\n" + "=" * 50)
    if all_passed:
        print("🎉 CLOUD GPU SETUP COMPLETE!")
        print("💡 Your cloud GPU is ready for Qwen model!")
        print("\\n🚀 Next steps:")
        print("  1. Run: python test_qwen_cloud.py")
        print("  2. Or start Streamlit: streamlit run launch.py --server.address=0.0.0.0 --server.port=8501")
    else:
        print("⚠️  Setup incomplete. Check errors above.")
        print("💡 Try restarting and running this script again.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()