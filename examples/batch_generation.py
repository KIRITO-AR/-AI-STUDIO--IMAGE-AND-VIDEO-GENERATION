"""
Batch generation example for AI Generation Studio.
Demonstrates generating multiple images from a list of prompts.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Default generation parameter constants
DEFAULT_IMAGE_WIDTH = 512
DEFAULT_IMAGE_HEIGHT = 512
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_IMAGES_PER_PROMPT = 1

from core import get_generation_engine, get_model_manager, GenerationParams

def main():
    print("🔄 AI Generation Studio - Batch Generation Example")
    print("=" * 60)
    
    # List of prompts to generate
    prompts = [
        "a majestic dragon flying over a medieval castle",
        "a serene lake surrounded by autumn trees",
        "a futuristic city with flying cars at night",
        "a cozy library with warm lighting and books",
        "a space explorer on an alien planet with two moons"
    ]
    
    # Initialize components
    print("Initializing components...")
    model_manager = get_model_manager()
    generation_engine = get_generation_engine()
    
    # Load model
    print("Loading model...")
    if not model_manager.load_model("sd15"):
        print("❌ Failed to load model")
        return
    
    print("✅ Model loaded successfully!")
    
    # Create base parameters
    base_params = GenerationParams(
        prompt="",  # Will be set for each generation
        negative_prompt="blurry, low quality, deformed",
        width=DEFAULT_IMAGE_WIDTH,
        height=DEFAULT_IMAGE_HEIGHT,
        num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale=DEFAULT_GUIDANCE_SCALE,
        num_images_per_prompt=DEFAULT_NUM_IMAGES_PER_PROMPT
    )
    
    print(f"Starting batch generation for {len(prompts)} prompts...")
    
    # Progress callback
    def progress_callback(current, total):
        progress = (current / total) * 100
        print(f"Progress: {current}/{total} ({progress:.1f}%)")
    
    # Generate batch
    results = generation_engine.batch_generate(prompts, base_params, progress_callback)
    
    # Process results
    successful = [r for r in results if not r.error]
    failed = [r for r in results if r.error]
    
    print("\n📊 Batch Generation Results:")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if failed:
        print("\nFailed generations:")
        for result in failed:
            print(f"  - '{result.params.prompt[:40]}...': {result.error}")
    
    # Save successful results
    if successful:
        print("\nSaving images...")
        total_saved = 0
        for i, result in enumerate(successful):
            saved_paths = result.save_images(
                "outputs/examples/batch",
                f"batch_{i:03d}"
            )
            total_saved += len(saved_paths)
            print(f"  Saved: {saved_paths[0]}")
        
        print(f"\n✅ Batch generation completed!")
        print(f"📁 Saved {total_saved} images to outputs/examples/batch/")

if __name__ == "__main__":
    main()