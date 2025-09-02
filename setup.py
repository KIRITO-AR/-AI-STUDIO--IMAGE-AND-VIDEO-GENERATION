"""
Setup script for AI Generation Studio.
Handles initial setup, dependency checking, and model downloads.
"""

import sys
import subprocess
import os
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version}")
    return True

def check_gpu():
    """Check for GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA GPU detected: {gpu_name} ({gpu_count} GPU(s))")
            return True
        else:
            print("⚠️  No CUDA GPU detected - will use CPU (slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        # Install with pip
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "outputs",
        "outputs/examples",
        "outputs/batch",
        "models_cache",
        "logs"
    ]
    
    print("\n📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {directory}")

def test_basic_functionality():
    """Test basic functionality."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test imports
        sys.path.append(str(Path(__file__).parent / "src"))
        
        from utils.gpu_utils import get_device_info
        from utils.config import get_config
        
        # Test GPU detection
        device, gpu_detector = get_device_info()
        print(f"  Device detected: {device}")
        
        # Test configuration
        config = get_config()
        print(f"  Configuration loaded successfully")
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def show_next_steps():
    """Show next steps to the user."""
    print("\n🎉 Setup completed successfully!")
    print("\n📖 Next steps:")
    print("1. Launch the Streamlit interface:")
    print("   python launch.py --streamlit")
    print()
    print("2. Or try the examples:")
    print("   python examples/simple_generation.py")
    print("   python examples/batch_generation.py")
    print()
    print("3. Open your browser to: http://localhost:8501")
    print()
    print("💡 Tips:")
    print("- First run will download models (this may take a while)")
    print("- CUDA GPU recommended for best performance")
    print("- Check the docs/ folder for more information")

def main():
    """Main setup function."""
    print("🚀 AI Generation Studio Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check for existing installation
    if Path("venv").exists():
        print("📦 Virtual environment detected")
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation")
        return
    
    # Create directories
    create_directories()
    
    # Test functionality
    if not test_basic_functionality():
        print("\n⚠️  Setup completed but basic tests failed")
        print("You may need to check your installation")
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main()