#!/usr/bin/env python3
"""
Test script for FLUX model integration.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_flux_model():
    """Test FLUX model loading and generation."""
    try:
        logger.info("Testing FLUX model integration...")
        
        # Import our modules
        from core.model_manager import ModelManager
        from core.generation_engine import GenerationEngine, GenerationParams
        
        # Initialize model manager
        logger.info("Initializing model manager...")
        model_manager = ModelManager()
        
        # Check available models
        logger.info("Available models:")
        for model in model_manager.list_models():
            logger.info(f"  - {model.name} ({model.model_id})")
        
        # Check FLUX model compatibility
        logger.info("Checking FLUX model compatibility...")
        flux_compat = model_manager.check_model_compatibility("flux_dev")
        logger.info(f"FLUX.1-dev compatibility: {flux_compat}")
        
        if flux_compat.get("compatible", False):
            logger.info("Loading FLUX.1-dev model...")
            if model_manager.load_model("flux_dev"):
                logger.info("FLUX.1-dev model loaded successfully!")
                
                # Test generation
                logger.info("Testing generation...")
                engine = GenerationEngine()
                
                params = GenerationParams(
                    prompt="A cat holding a sign that says hello world",
                    width=1024,
                    height=1024,
                    num_inference_steps=50,
                    guidance_scale=3.5,
                    seed=0
                )
                
                result = engine.generate_image(params)
                if result.error:
                    logger.error(f"Generation failed: {result.error}")
                else:
                    logger.info(f"Generation successful in {result.generation_time:.2f}s")
                    # Save the image
                    output_dir = current_dir / "outputs"
                    output_dir.mkdir(exist_ok=True)
                    saved_paths = result.save_images(str(output_dir), "flux_test")
                    logger.info(f"Images saved to: {saved_paths}")
            else:
                logger.error("Failed to load FLUX.1-dev model")
        else:
            logger.warning(f"FLUX.1-dev not compatible: {flux_compat.get('reason', 'Unknown')}")
            logger.info("Recommendations:")
            for rec in flux_compat.get('recommendations', []):
                logger.info(f"  - {rec}")
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_flux_model()