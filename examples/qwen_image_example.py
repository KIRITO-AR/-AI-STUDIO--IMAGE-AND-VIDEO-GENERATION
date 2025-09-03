#!/usr/bin/env python3
"""
Qwen-Image Generation Example

This script demonstrates how to use the Qwen-Image model for high-quality image generation
with multiple aspect ratios and multilingual support.

Requirements:
- 8GB+ GPU VRAM (or CPU mode for slower generation)
- Hugging Face account with Qwen-Image access
- diffusers, torch, and transformers installed
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir))

from core.model_manager import get_model_manager
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Qwen-Image specific settings
POSITIVE_MAGIC = {
    "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
    "zh": ", 超清，4K，电影级构图."  # for chinese prompt
}

# Supported aspect ratios for Qwen-Image
ASPECT_RATIOS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

def generate_qwen_image(
    prompt: str,
    negative_prompt: str = "",
    aspect_ratio: str = "16:9",
    num_inference_steps: int = 50,
    true_cfg_scale: float = 4.0,
    seed: int = 42,
    output_path: str = "qwen_output.png"
):
    """
    Generate an image using Qwen-Image model.
    
    Args:
        prompt: Text description of the image to generate
        negative_prompt: What to avoid in the image (optional)
        aspect_ratio: Image aspect ratio (default: "16:9")
        num_inference_steps: Number of denoising steps (default: 50)
        true_cfg_scale: Guidance scale for generation (default: 4.0)
        seed: Random seed for reproducibility (default: 42)
        output_path: Where to save the generated image
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize model manager
        model_manager = get_model_manager()
        
        # Check if Qwen model is compatible
        compatibility = model_manager.check_model_compatibility("qwen_image")
        if not compatibility["compatible"]:
            logger.error(f"Qwen-Image not compatible: {compatibility['reason']}")
            logger.info("Recommendations:")
            for rec in compatibility.get("recommendations", []):
                logger.info(f"  - {rec}")
            return False
        
        # Load Qwen-Image model
        logger.info("Loading Qwen-Image model...")
        if not model_manager.load_model("qwen_image"):
            logger.error("Failed to load Qwen-Image model")
            return False
        
        # Get the pipeline
        pipeline = model_manager.current_pipeline
        if pipeline is None:
            logger.error("Pipeline not available")
            return False
        
        # Get resolution for aspect ratio
        if aspect_ratio not in ASPECT_RATIOS:
            logger.warning(f"Unknown aspect ratio {aspect_ratio}, using 16:9")
            aspect_ratio = "16:9"
        
        width, height = ASPECT_RATIOS[aspect_ratio]
        logger.info(f"Generating {width}x{height} image with aspect ratio {aspect_ratio}")
        
        # Enhance prompt with positive magic
        enhanced_prompt = prompt + POSITIVE_MAGIC["en"]
        
        # Set up generation parameters
        generator = torch.Generator(device=model_manager.device).manual_seed(seed)
        
        # Generate image
        logger.info("Generating image...")
        result = pipeline(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=generator
        )
        
        # Save the image
        image = result.images[0]
        image.save(output_path)
        logger.info(f"Image saved to: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return False

def main():
    """Main example function."""
    # Example prompts
    prompts = [
        'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197"',
        "A serene mountain landscape at sunset with clouds reflecting in a crystal clear lake",
        "A futuristic cityscape with flying cars and neon lights at night",
        "A magical forest with glowing mushrooms and fairy lights"
    ]
    
    # Create output directory
    output_dir = Path("qwen_outputs")
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Starting Qwen-Image generation examples...")
    
    for i, prompt in enumerate(prompts):
        logger.info(f"\nGenerating image {i+1}/{len(prompts)}")
        logger.info(f"Prompt: {prompt[:80]}...")
        
        output_path = output_dir / f"qwen_example_{i+1}.png"
        
        success = generate_qwen_image(
            prompt=prompt,
            aspect_ratio="16:9",
            num_inference_steps=50,
            seed=42 + i,
            output_path=str(output_path)
        )
        
        if success:
            logger.info(f"✅ Successfully generated image {i+1}")
        else:
            logger.error(f"❌ Failed to generate image {i+1}")
    
    logger.info("\nQwen-Image generation examples completed!")

if __name__ == "__main__":
    main()
