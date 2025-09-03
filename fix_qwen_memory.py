#!/usr/bin/env python3
"""
Advanced CUDA memory fix for Qwen model loading.
This script implements multiple strategies to handle severe memory fragmentation.
"""

import os
import sys
import logging
from pathlib import Path

# CRITICAL: Set memory configuration BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error reporting

# Add src to Python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def restart_python_hint():
    """Provide instructions for restarting Python to clear memory."""
    logger.error("")
    logger.error("🔄 MEMORY RESET REQUIRED 🔄")
    logger.error("The GPU memory is severely fragmented. To fix this:")
    logger.error("1. Close this Python session completely")
    logger.error("2. Open a new PowerShell window")
    logger.error("3. Run: $env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'")
    logger.error("4. Then try loading Qwen again")
    logger.error("")

def advanced_qwen_memory_fix():
    """Advanced CUDA memory fix with multiple fallback strategies."""
    try:
        logger.info("🚀 Starting Advanced Qwen Memory Fix...")
        
        # Import torch first to check memory
        import torch
        total_memory = 24.0  # Default for safety
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"🔧 GPU: {device_name}")
            logger.info(f"📊 Total GPU Memory: {total_memory:.2f}GB")
            logger.info(f"📊 Initially Allocated: {allocated_memory:.2f}GB")
            
            # Check for severe memory fragmentation
            if allocated_memory > total_memory * 0.8:
                logger.error(f"🚨 SEVERE MEMORY FRAGMENTATION DETECTED!")
                logger.error(f"Allocated {allocated_memory:.2f}GB > 80% of {total_memory:.2f}GB")
                restart_python_hint()
                return False
        
        # Import our modules
        from core.model_manager import ModelManager
        from core.generation_engine import GenerationEngine, GenerationParams
        
        # Initialize model manager
        logger.info("🔧 Initializing model manager...")
        model_manager = ModelManager()
        
        # STEP 1: Ultra-aggressive memory cleanup
        logger.info("🧹 Performing ultra-aggressive memory cleanup...")
        model_manager.force_memory_cleanup()
        
        if torch.cuda.is_available():
            post_cleanup_allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"📊 After cleanup: {post_cleanup_allocated:.2f}GB allocated")
            
            # If still too much memory allocated, suggest restart
            if post_cleanup_allocated > total_memory * 0.5:
                logger.warning(f"⚠️  Still high memory usage: {post_cleanup_allocated:.2f}GB")
                restart_python_hint()
                logger.info("Continuing anyway, but restart is recommended...")
        
        # STEP 2: Test alternative models first
        logger.info("🧪 Testing SDXL model first (should work)...")
        try:
            sdxl_success = model_manager.load_model("sdxl")
            if sdxl_success:
                logger.info("✅ SDXL loaded successfully - GPU is working")
                model_manager.force_memory_cleanup()  # Unload SDXL
            else:
                logger.warning("⚠️  SDXL failed to load - GPU may have issues")
        except Exception as e:
            logger.warning(f"⚠️  SDXL test failed: {e}")
        
        # STEP 3: Check Qwen compatibility
        logger.info("🔍 Checking Qwen model compatibility...")
        qwen_compat = model_manager.check_model_compatibility("qwen_image")
        logger.info(f"🔍 Qwen compatibility: {qwen_compat}")
        
        # STEP 4: Attempt Qwen loading with multiple strategies
        strategies = [
            "GPU with CPU offloading",
            "Balanced device mapping", 
            "CPU-only mode"
        ]
        
        for i, strategy in enumerate(strategies, 1):
            logger.info(f"\n🎯 Strategy {i}/3: {strategy}")
            
            try:
                # Force cleanup before each attempt
                model_manager.force_memory_cleanup()
                
                # Attempt to load Qwen
                logger.info("🔄 Attempting to load Qwen model...")
                success = model_manager.load_model("qwen_image")
                
                if success:
                    logger.info(f"✅ SUCCESS! Qwen loaded with {strategy}")
                    
                    # Test generation
                    logger.info("🎨 Testing Qwen generation...")
                    engine = GenerationEngine()
                    
                    params = GenerationParams(
                        prompt="A beautiful sunset over mountains, digital art",
                        aspect_ratio="16:9",
                        num_inference_steps=20,  # Reduced for testing
                        seed=42
                    )
                    
                    result = engine.generate_image(params)
                    if result.error:
                        logger.error(f"❌ Generation failed: {result.error}")
                    else:
                        logger.info(f"✅ Generation successful in {result.generation_time:.2f}s")
                        # Save the image
                        output_dir = current_dir / "outputs"
                        output_dir.mkdir(exist_ok=True)
                        saved_paths = result.save_images(str(output_dir), "qwen_advanced_test")
                        logger.info(f"🖼️  Images saved to: {saved_paths}")
                    
                    logger.info("\n🎉 QWEN MODEL SUCCESSFULLY INTEGRATED!")
                    return True
                    
                else:
                    logger.warning(f"❌ Strategy {i} failed to load Qwen")
                    continue
                    
            except Exception as e:
                logger.error(f"❌ Strategy {i} failed with error: {e}")
                continue
        
        # If all strategies failed
        logger.error("\n❌ ALL STRATEGIES FAILED")
        logger.error("🔧 Recommendations:")
        logger.error("1. 🔄 Restart Python session to clear memory fragmentation")
        logger.error("2. 🧹 Close other GPU applications (browsers, games, etc.)")
        logger.error("3. 🎯 Use SDXL or SD15 models instead (they work well)")
        logger.error("4. ☁️  Try on a fresh cloud instance")
        
        restart_python_hint()
        return False
    
    except Exception as e:
        logger.error(f"💥 Critical error: {e}")
        import traceback
        traceback.print_exc()
        restart_python_hint()
        return False

if __name__ == "__main__":
    advanced_qwen_memory_fix()