#!/usr/bin/env python3
"""
Vultr Cloud GPU Startup Script for AI Generation Studio
Optimized for 64GB VRAM cloud instances
"""

import traceback
import sys
import os
import subprocess
from pathlib import Path

def check_gpu():
    """Check if GPU is available and display info."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU Detected: {gpu_name}")
            print(f"✅ VRAM: {gpu_memory:.1f}GB")
            
            if gpu_memory >= 48:
                print("🚀 Cloud GPU detected! Optimizing for high-performance generation...")
                return True
            elif gpu_memory >= 24:
                print("⚡ High-end GPU detected! Enabling advanced models...")
                return True
            else:
                print("💡 Standard GPU detected.")
                return True
        else:
            print("❌ No CUDA GPU detected. Please check your setup.")
            return False
    except ImportError:
        print("❌ PyTorch not installed. Please install requirements.")
        return False

def check_network_access():
    """Check if the instance allows external access."""
    print("\n🌐 Network Configuration:")
    print("   Make sure your Vultr instance has:")
    print("   - Port 8501 open in firewall")
    print("   - Public IP accessible")
    print("   - Security groups configured properly")

def display_access_info():
    """Display access information for the cloud instance."""
    print("\n📡 Cloud GPU Access Information:")
    print("   Local access: http://localhost:8501")
    print("   Remote access: http://YOUR_VULTR_IP:8501")
    print("   Replace YOUR_VULTR_IP with your actual Vultr instance IP")

def main():
    """Main startup function."""
    print("=" * 60)
    print("🎨 AI Generation Studio - Vultr Cloud GPU Setup")
    print("=" * 60)
    
    # Check GPU
    if not check_gpu():
        sys.exit(1)
    
    # Display network info
    check_network_access()
    
    # Display access info
    display_access_info()
    
    # Launch the application
    print("\n🚀 Starting AI Generation Studio...")
    print("   Press Ctrl+C to stop the application")
    
    try:
        # Change to project directory
        project_dir = Path(__file__).parent
        os.chdir(project_dir)
        
        # Launch with streamlit, ensuring the subprocess is properly managed
        process = subprocess.Popen([sys.executable, "launch.py", "--streamlit"])
        try:
            process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.SubprocessError as e:
        print(f"\n❌ Error starting application: {e}")
        traceback.print_exc()
        sys.exit(1)
    except OSError as e:
        print(f"\n❌ Error starting application: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()