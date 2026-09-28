"""
Simple example of using the AI Generation Studio API.
This script demonstrates basic image generation.
"""

import sys
from pathlib import Path

# Default generation parameters
DEFAULT_IMAGE_WIDTH = 512
DEFAULT_IMAGE_HEIGHT = 512
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_IMAGES_PER_PROMPT = 1

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core import get_generation_engine, get_model_manager, GenerationParams

def main():
    print("🎨 AI Generation Studio - Simple Example")
    print("=" * 50)
    
    # Initialize components
    print("Initializing components...")
    model_manager = get_model_manager()
    generation_engine = get_generation_engine()
    
    # Load a model
    print("Loading Stable Diffusion 1.5...")
    if not model_manager.load_model("sd15"):
        print("❌ Failed to load model")
        return
    
    print("✅ Model loaded successfully!")
    
    # Create generation parameters
    params = GenerationParams(
        prompt="a beautiful sunset over mountains, digital art, highly detailed",
        negative_prompt="blurry, low quality",
        width=DEFAULT_IMAGE_WIDTH,
        height=DEFAULT_IMAGE_HEIGHT,
        num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale=DEFAULT_GUIDANCE_SCALE,
        num_images_per_prompt=DEFAULT_NUM_IMAGES_PER_PROMPT
    )
    
    print(f"Generating image with prompt: '{params.prompt}'")
    
    # Generate image
    result = generation_engine.generate_image(params)
    
    if result.error:
        print(f"❌ Generation failed: {result.error}")
        return
    
    # Save the result
    saved_paths = result.save_images("outputs/examples", "simple_example")
    
    print(f"✅ Generation completed in {result.generation_time:.2f}s")
    print(f"📁 Saved to: {saved_paths[0]}")
    print(f"🎲 Seed used: {result.seed_used}")

if __name__ == "__main__":
    main()