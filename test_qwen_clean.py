#!/usr/bin/env python3
"""
Test Qwen-Image Model

This script tests the Qwen-Image model integration to ensure it works correctly.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

from core.model_manager import get_model_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_qwen_model():
    """Test Qwen-Image model loading and basic functionality."""
    try:
        logger.info("🧪 Testing Qwen-Image model...")
        
        # Initialize model manager
        model_manager = get_model_manager()
        
        # Test 1: Check model registry
        logger.info("📋 Test 1: Checking model registry...")
        qwen_info = model_manager.get_model_info("qwen_image")
        if qwen_info:
            logger.info(f"✅ Model found: {qwen_info.name}")
            logger.info(f"   Model ID: {qwen_info.model_id}")
            logger.info(f"   Memory requirement: {qwen_info.memory_requirement}MB")
            logger.info(f"   Description: {qwen_info.description}")
        else:
            logger.error("❌ Qwen model not found in registry")
            return False
        
        # Test 2: Check compatibility
        logger.info("\n🔍 Test 2: Checking hardware compatibility...")
        compatibility = model_manager.check_model_compatibility("qwen_image")
        logger.info(f"Compatible: {compatibility['compatible']}")
        logger.info(f"Reason: {compatibility['reason']}")
        if compatibility.get('recommendations'):
            logger.info("Recommendations:")
            for rec in compatibility['recommendations']:
                logger.info(f"  - {rec}")
        
        if not compatibility['compatible']:
            logger.warning("⚠️  Hardware not fully compatible, but test will continue")
        
        # Test 3: Attempt to load model
        logger.info("\n⚙️  Test 3: Loading Qwen-Image model...")
        load_success = model_manager.load_model("qwen_image")
        
        if load_success:
            logger.info("✅ Model loaded successfully!")
            
            # Test 4: Check if pipeline is available
            current_model = model_manager.get_current_model()
            if current_model and current_model.model_id == "Qwen/Qwen-Image":
                logger.info("✅ Pipeline is ready for generation")
                
                # Cleanup
                model_manager.cleanup()
                logger.info("🧹 Cleanup completed")
                
                return True
            else:
                logger.error("❌ Pipeline not properly initialized")
                return False
        else:
            logger.error("❌ Failed to load model")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

def main():
    """Run the test suite."""
    logger.info("🚀 Starting Qwen-Image model test suite...")
    
    success = test_qwen_model()
    
    if success:
        logger.info("🎉 All tests passed! Qwen-Image model is ready to use.")
        logger.info("\n💡 Next steps:")
        logger.info("   1. Run examples/qwen_image_example.py for full generation example")
        logger.info("   2. Use the model in your own scripts")
        logger.info("   3. Experiment with different prompts and aspect ratios")
    else:
        logger.error("💥 Tests failed. Please check the logs above for issues.")
        logger.info("\n🔧 Troubleshooting:")
        logger.info("   1. Ensure you have sufficient GPU memory (8GB+)")
        logger.info("   2. Check your Hugging Face access to Qwen/Qwen-Image")
        logger.info("   3. Verify all dependencies are installed")
        logger.info("   4. Try running: huggingface-cli login")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
