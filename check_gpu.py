#!/usr/bin/env python3
"""
Script to check GPU detection and get memory recommendations.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    print("=== GPU Detection Test ===")
    
    # Import after path setup
    from utils.gpu_utils import get_device_info
    
    device, detector = get_device_info()
    
    print(f"Device: {device}")
    print(f"CUDA available: {detector.cuda_available}")
    print(f"ROCm available: {detector.rocm_available}")
    print(f"GPU count: {len(detector.gpus)}")
    
    if detector.gpus:
        for i, gpu in enumerate(detector.gpus):
            print(f"\nGPU {i}:")
            print(f"  Name: {gpu.name}")
            print(f"  Memory: {gpu.memory_total}MB total, {gpu.memory_free}MB free")
            print(f"  Utilization: {gpu.utilization}%")
            if gpu.temperature:
                print(f"  Temperature: {gpu.temperature}°C")
            if gpu.driver_version:
                print(f"  Driver: {gpu.driver_version}")
        
        best_gpu = detector.get_best_gpu()
        if best_gpu:
            print(f"\nBest GPU: {best_gpu.name}")
            
            recommendations = detector.get_memory_recommendations()
            print("\nMemory Recommendations:")
            for key, value in recommendations.items():
                print(f"  {key}: {value}")
            
            optimizations = detector.get_optimization_settings()
            print("\nOptimization Settings:")
            for key, value in optimizations.items():
                print(f"  {key}: {value}")
    
    # System info
    system_info = detector.system_info
    print(f"\nSystem Info:")
    print(f"  CPUs: {system_info.cpu_count}")
    print(f"  RAM: {system_info.ram_available}GB available / {system_info.ram_total}GB total")
    print(f"  Platform: {system_info.platform}")
    
    # Check PyTorch optimization environment
    print(f"\nPyTorch Memory Settings:")
    print(f"  PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')}")

if __name__ == "__main__":
    main()