#!/usr/bin/env python3
"""
Simple Qwen-Image Test

Direct test of Qwen-Image model using the clean implementation from your example.
"""

import logging
import torch
from diffusers import DiffusionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_qwen_direct():
    """Test Qwen-Image model directly using the clean approach."""
    try:
        logger.info("🧪 Testing Qwen-Image model directly...")
        
        model_name = "Qwen/Qwen-Image"
        
        # Determine device and dtype
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
            device = "cuda"
            logger.info(f"🚀 Using CUDA with bfloat16 precision")
        else:
            torch_dtype = torch.float32
            device = "cpu"
            logger.info(f"💻 Using CPU with float32 precision")
        
        logger.info(f"Loading model: {model_name}")
        
        # Load the pipeline (clean approach from your example)
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
        pipe = pipe.to(device)
        
        logger.info("✅ Model loaded successfully!")
        
        # Test generation parameters
        positive_magic = {
            "en": ", Ultra HD, 4K, cinematic composition.",
            "zh": ", 超清，4K，电影级构图."
        }
        
        # Test aspect ratios
        aspect_ratios = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1140),
            "3:4": (1140, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        }
        
        # Simple test prompt
        prompt = "A beautiful sunset over mountains"
        negative_prompt = ""
        width, height = aspect_ratios["16:9"]
        
        logger.info(f"🎨 Testing generation with prompt: {prompt}")
        logger.info(f"📐 Resolution: {width}x{height}")
        
        # Generate test image
        image = pipe(
            prompt=prompt + positive_magic["en"],
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=20,  # Fewer steps for testing
            true_cfg_scale=4.0,
            generator=torch.Generator(device=device).manual_seed(42)
        ).images[0]
        
        # Save test image
        output_path = "qwen_test_output.png"
        image.save(output_path)
        logger.info(f"💾 Test image saved to: {output_path}")
        
        logger.info("🎉 All tests passed! Qwen-Image model is working correctly.")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("💡 Make sure you have installed: pip install diffusers torch transformers")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        
        # Check for common issues
        error_str = str(e).lower()
        if 'out of memory' in error_str:
            logger.info("💡 GPU memory issue. Try:")
            logger.info("   1. Close other applications")
            logger.info("   2. Use smaller resolution")
            logger.info("   3. Set torch_dtype=torch.float16")
        elif 'access' in error_str or 'permission' in error_str:
            logger.info("💡 Access issue. Try:")
            logger.info("   1. Accept model license on Hugging Face")
            logger.info("   2. Run: huggingface-cli login")
        elif 'connection' in error_str or 'network' in error_str:
            logger.info("💡 Network issue. Check internet connection")
        
        return False

def main():
    """Run the direct test."""
    logger.info("🚀 Starting direct Qwen-Image test...")
    
    success = test_qwen_direct()
    
    if success:
        logger.info("✅ Direct test completed successfully!")
        logger.info("🎯 The clean Qwen-Image implementation is working correctly.")
    else:
        logger.error("❌ Direct test failed.")
        logger.info("🔧 Please check the error messages above for troubleshooting.")
    
    return success

if __name__ == "__main__":
    main()
