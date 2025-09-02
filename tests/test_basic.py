"""
Basic tests for AI Generation Studio components.
"""

import sys
import pytest
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all core modules can be imported."""
    try:
        from core import get_model_manager, get_generation_engine, GenerationParams
        from utils import get_config, get_device_info
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_config_loading():
    """Test configuration loading."""
    from utils import get_config
    
    config = get_config()
    assert config is not None
    assert hasattr(config, 'models')
    assert hasattr(config, 'generation')

def test_gpu_detection():
    """Test GPU detection."""
    from utils import get_device_info
    
    device, gpu_detector = get_device_info()
    assert device in ['cpu', 'cuda:0', 'cuda:1']  # Basic device options
    assert gpu_detector is not None

def test_model_manager_init():
    """Test model manager initialization."""
    from core import get_model_manager
    
    model_manager = get_model_manager()
    assert model_manager is not None
    assert len(model_manager.available_models) > 0

def test_generation_params():
    """Test generation parameters."""
    from core import GenerationParams
    
    params = GenerationParams(
        prompt="test prompt",
        width=512,
        height=512
    )
    
    assert params.prompt == "test prompt"
    assert params.width == 512
    assert params.height == 512
    assert params.num_inference_steps == 20  # default

def test_generation_params_validation():
    """Test parameter validation."""
    from core import GenerationParams
    
    # Test width/height limits
    params = GenerationParams(
        prompt="test",
        width=5000,  # Too large
        height=10    # Too small
    )
    
    assert params.width <= 2048
    assert params.height >= 64

if __name__ == "__main__":
    # Run tests directly
    import unittest
    
    class TestBasicFunctionality(unittest.TestCase):
        
        def test_imports(self):
            test_imports()
        
        def test_config_loading(self):
            test_config_loading()
        
        def test_gpu_detection(self):
            test_gpu_detection()
        
        def test_model_manager_init(self):
            test_model_manager_init()
        
        def test_generation_params(self):
            test_generation_params()
        
        def test_generation_params_validation(self):
            test_generation_params_validation()
    
    unittest.main()