#!/usr/bin/env python3
"""
Fix CUDA out of memory issues for Qwen model loading.
"""

import os
import sys
import logging
from pathlib import Path

# Set PyTorch memory allocation configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add src to Python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_qwen_memory_issues():
    """Fix CUDA out of memory issues and test Qwen model loading."""
    try:
        logger.info("Starting CUDA memory fix for Qwen model...")
        
        # Import torch first to set environment variables
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA Device: {torch.cuda.get_device_name()}")
            logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
            logger.info(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            logger.info(f"Free Memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.2f}GB")
        
        # Import our modules
        from core.model_manager import ModelManager
        from core.generation_engine import GenerationEngine, GenerationParams
        
        # Initialize model manager
        logger.info("Initializing model manager...")
        model_manager = ModelManager()
        
        # Force aggressive memory cleanup first
        logger.info("Performing aggressive memory cleanup...")
        model_manager.force_memory_cleanup()
        
        if torch.cuda.is_available():
            logger.info(f"After cleanup - Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            logger.info(f"After cleanup - Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.2f}GB")
        
        # Check Qwen model compatibility
        logger.info("Checking Qwen model compatibility...")
        qwen_compat = model_manager.check_model_compatibility("qwen_image")
        logger.info(f"Qwen-Image compatibility: {qwen_compat}")
        
        if qwen_compat.get("compatible", False):
            logger.info("Attempting to load Qwen model...")
            success = model_manager.load_model("qwen_image")
            
            if success:
                logger.info("✅ Qwen model loaded successfully!")
                
                # Test generation
                logger.info("Testing Qwen generation...")
                engine = GenerationEngine()
                
                params = GenerationParams(
                    prompt="A coffee shop entrance features a chalkboard sign reading 'Qwen Coffee 😊 $2 per cup'",
                    aspect_ratio="16:9",
                    num_inference_steps=30,  # Reduced steps for testing
                    seed=42
                )
                
                result = engine.generate_image(params)
                if result.error:
                    logger.error(f"Generation failed: {result.error}")
                else:
                    logger.info(f"✅ Generation successful in {result.generation_time:.2f}s")
                    # Save the image
                    output_dir = current_dir / "outputs"
                    output_dir.mkdir(exist_ok=True)
                    saved_paths = result.save_images(str(output_dir), "qwen_test")
                    logger.info(f"Images saved to: {saved_paths}")
            else:
                logger.error("❌ Failed to load Qwen model")
                logger.info("Suggestions:")
                logger.info("1. Ensure you have enough GPU memory (8GB+ required)")
                logger.info("2. Close other applications using GPU")
                logger.info("3. Try restarting the Python process")
                logger.info("4. Use SDXL or SD15 models instead")
        else:
            logger.warning(f"❌ Qwen model not compatible: {qwen_compat.get('reason', 'Unknown')}")
            logger.info("Recommendations:")
            for rec in qwen_compat.get('recommendations', []):
                logger.info(f"  - {rec}")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_qwen_memory_issues()