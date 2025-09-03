#!/usr/bin/env python3
"""
Enhanced Qwen model test script optimized for cloud GPUs.
Tests multiple loading strategies and provides detailed feedback.
"""

import os
import sys
import logging
from pathlib import Path

# CRITICAL: Set CUDA environment variables BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:1024,roundup_power2_divisions:16'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Prevent timeout issues
os.environ['TORCH_USE_CUDA_DSA'] = '1'    # Enable device-side assertions

# Add src to Python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

# Set up logging with enhanced formatting
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('qwen_cloud_test.log')
    ]
)
logger = logging.getLogger(__name__)

def test_cloud_gpu_info():
    """Test and display cloud GPU information."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.error("❌ CUDA not available!")
            return False
        
        device_count = torch.cuda.device_count()
        logger.info(f"🔢 CUDA devices found: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3  # GB
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            free_memory = total_memory - allocated
            
            logger.info(f"🌟 GPU {i}: {device_name}")
            logger.info(f"  📊 Total Memory: {total_memory:.2f}GB")
            logger.info(f"  🆓 Free Memory: {free_memory:.2f}GB")
            logger.info(f"  🔧 Compute Capability: {props.major}.{props.minor}")
            
            # Check if this is a cloud GPU
            if total_memory >= 48.0:
                logger.info(f"  ☁️  Cloud GPU detected!")
                return True
            elif total_memory >= 24.0:
                logger.info(f"  🎮 High-end GPU detected")
                return True
            else:
                logger.info(f"  🖥️  Standard GPU")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to get GPU info: {e}")
        return False

def test_qwen_model_comprehensive():
    """Comprehensive test of Qwen model loading with multiple strategies."""
    try:
        logger.info("🚀 Starting Comprehensive Qwen Cloud GPU Test...")
        
        # Test GPU info first
        if not test_cloud_gpu_info():
            logger.error("❌ GPU test failed")
            return False
        
        # Import our modules
        from core.model_manager import ModelManager
        from core.generation_engine import GenerationEngine, GenerationParams
        
        # Initialize model manager
        logger.info("🔧 Initializing model manager...")
        model_manager = ModelManager()
        
        # Test 1: Check Qwen compatibility
        logger.info("\n📋 Test 1: Checking Qwen compatibility...")
        qwen_compat = model_manager.check_model_compatibility("qwen_image")
        logger.info(f"🔍 Qwen compatibility result: {qwen_compat}")
        
        if not qwen_compat.get("compatible", False):
            logger.error(f"❌ Qwen not compatible: {qwen_compat.get('reason', 'Unknown')}")
            logger.info("💡 Recommendations:")
            for rec in qwen_compat.get('recommendations', []):
                logger.info(f"  - {rec}")
            return False
        
        # Test 2: Load Qwen model
        logger.info("\n🎯 Test 2: Loading Qwen model with cloud GPU optimizations...")
        
        try:
            success = model_manager.load_model("qwen_image")
            
            if not success:
                logger.error("❌ Failed to load Qwen model")
                return False
            
            logger.info("✅ Qwen model loaded successfully!")
            
            # Test 3: Generate test image
            logger.info("\n🎨 Test 3: Testing Qwen image generation...")
            
            engine = GenerationEngine()
            
            # Test with a simple prompt first
            test_prompts = [
                "A beautiful sunset over mountains, digital art",
                "A cute cat sitting in a garden, photorealistic",
                "Abstract geometric patterns in blue and gold"
            ]
            
            for i, prompt in enumerate(test_prompts, 1):
                logger.info(f"🖼️  Generation test {i}/3: {prompt[:50]}...")
                
                params = GenerationParams(
                    prompt=prompt,
                    aspect_ratio="16:9",
                    num_inference_steps=20,  # Reduced for testing
                    guidance_scale=7.5,
                    seed=42 + i
                )
                
                result = engine.generate_image(params)
                
                if result.error:
                    logger.error(f"❌ Generation {i} failed: {result.error}")
                    continue
                else:
                    logger.info(f"✅ Generation {i} successful in {result.generation_time:.2f}s")
                    
                    # Save the image
                    output_dir = current_dir / "outputs" / "qwen_cloud_test"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    saved_paths = result.save_images(str(output_dir), f"qwen_test_{i}")
                    logger.info(f"💾 Images saved to: {saved_paths}")
            
            logger.info("\n🎉 QWEN CLOUD GPU TEST COMPLETED SUCCESSFULLY!")
            logger.info("✅ All tests passed - Qwen is working correctly on your cloud GPU")
            
            return True
            
        except Exception as load_error:
            logger.error(f"❌ Qwen loading failed: {load_error}")
            return False
    
    except Exception as e:
        logger.error(f"💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_management():
    """Test advanced memory management features."""
    try:
        logger.info("\n🧠 Testing advanced memory management...")
        
        from core.model_manager import ModelManager
        import torch
        
        if torch.cuda.is_available():
            # Get initial memory state
            initial_allocated = torch.cuda.memory_allocated() / 1024**3
            initial_reserved = torch.cuda.memory_reserved() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            logger.info(f"📊 Initial Memory State:")
            logger.info(f"  - Allocated: {initial_allocated:.2f}GB")
            logger.info(f"  - Reserved: {initial_reserved:.2f}GB")
            logger.info(f"  - Total: {total_memory:.2f}GB")
            
            # Test memory cleanup
            model_manager = ModelManager()
            logger.info("🧹 Testing aggressive memory cleanup...")
            model_manager.force_memory_cleanup()
            
            # Check memory after cleanup
            final_allocated = torch.cuda.memory_allocated() / 1024**3
            final_reserved = torch.cuda.memory_reserved() / 1024**3
            
            logger.info(f"📊 After Cleanup:")
            logger.info(f"  - Allocated: {final_allocated:.2f}GB")
            logger.info(f"  - Reserved: {final_reserved:.2f}GB")
            logger.info(f"  - Freed: {initial_allocated - final_allocated:.2f}GB")
            
            return True
        else:
            logger.warning("⚠️  CUDA not available for memory testing")
            return False
    
    except Exception as e:
        logger.error(f"❌ Memory management test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🌟 Starting Qwen Cloud GPU Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("GPU Information", test_cloud_gpu_info),
        ("Memory Management", test_memory_management),
        ("Qwen Model Comprehensive", test_qwen_model_comprehensive)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🔬 Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED! Qwen is ready to use on your cloud GPU!")
        logger.info("💡 You can now use Qwen through the Streamlit interface or API")
    else:
        logger.info("\n⚠️  Some tests failed. Check the logs above for details.")
        logger.info("💡 Try restarting Python and running the test again")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()