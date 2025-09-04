"""
Video generation example for AI Generation Studio.
This script demonstrates advanced video generation capabilities.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core import get_generation_engine, get_model_manager, GenerationParams

def generate_basic_video():
    """Generate a basic video using the default video model."""
    print("🎬 AI Generation Studio - Video Generation Example")
    print("=" * 50)
    
    # Initialize components
    print("Initializing components...")
    model_manager = get_model_manager()
    generation_engine = get_generation_engine()
    
    # Load a video model
    print("Loading AnimateDiff video model...")
    if not model_manager.load_model("animatediff"):
        print("❌ Failed to load video model")
        return
    
    print("✅ Video model loaded successfully!")
    
    # Create generation parameters
    params = GenerationParams(
        prompt="a beautiful sunset over mountains, cinematic, highly detailed",
        negative_prompt="blurry, low quality, distorted",
        width=512,
        height=512,
        num_inference_steps=20,
        guidance_scale=7.5,
        num_frames=16,
        fps=8
    )
    
    print(f"Generating video with prompt: '{params.prompt}'")
    
    # Generate video
    result = generation_engine.generate_video(params)
    
    if result.error:
        print(f"❌ Video generation failed: {result.error}")
        return
    
    # Save the result
    saved_paths = result.save_images("outputs/examples", "video_example")
    
    # Save as video file
    try:
        video_path = generation_engine.save_result_as_video(
            result, "outputs/examples/generated_video.mp4"
        )
        print(f"🎬 Video saved to: {video_path}")
    except Exception as e:
        print(f"⚠️  Failed to save video file: {e}")
    
    print(f"✅ Video generation completed in {result.generation_time:.2f}s")
    print(f"📁 Individual frames saved to: {saved_paths[0]}")
    print(f"🎲 Seed used: {result.seed_used}")

def generate_modelscope_video():
    """Generate a video using ModelScope T2V model."""
    print("\n" + "=" * 50)
    print("🎬 Generating video with ModelScope T2V")
    print("=" * 50)
    
    # Initialize components
    model_manager = get_model_manager()
    generation_engine = get_generation_engine()
    
    # Load ModelScope T2V model
    print("Loading ModelScope T2V model...")
    if not model_manager.load_model("modelscope_t2v"):
        print("❌ Failed to load ModelScope T2V model")
        return
    
    print("✅ ModelScope T2V model loaded successfully!")
    
    # Create generation parameters
    params = GenerationParams(
        prompt="a futuristic city with flying cars, cyberpunk style, cinematic lighting",
        negative_prompt="blurry, low quality, static",
        width=512,
        height=512,
        num_inference_steps=25,
        guidance_scale=9.0,
        num_frames=24,
        fps=8
    )
    
    print(f"Generating video with prompt: '{params.prompt}'")
    
    # Generate video
    result = generation_engine.generate_video(params)
    
    if result.error:
        print(f"❌ Video generation failed: {result.error}")
        return
    
    # Save the result
    saved_paths = result.save_images("outputs/examples", "modelscope_video")
    
    # Save as video file
    try:
        video_path = generation_engine.save_result_as_video(
            result, "outputs/examples/modelscope_video.mp4"
        )
        print(f"🎬 Video saved to: {video_path}")
    except Exception as e:
        print(f"⚠️  Failed to save video file: {e}")
    
    print(f"✅ Video generation completed in {result.generation_time:.2f}s")
    print(f"📁 Individual frames saved to: {saved_paths[0]}")
    print(f"🎲 Seed used: {result.seed_used}")

def generate_zeroscope_video():
    """Generate a video using Zeroscope model."""
    print("\n" + "=" * 50)
    print("🎬 Generating video with Zeroscope")
    print("=" * 50)
    
    # Initialize components
    model_manager = get_model_manager()
    generation_engine = get_generation_engine()
    
    # Load Zeroscope model
    print("Loading Zeroscope model...")
    if not model_manager.load_model("zeroscope_v2_576w"):
        print("❌ Failed to load Zeroscope model")
        return
    
    print("✅ Zeroscope model loaded successfully!")
    
    # Create generation parameters
    params = GenerationParams(
        prompt="an astronaut riding a horse on mars, high quality, detailed",
        negative_prompt="blurry, low quality, distorted",
        width=576,
        height=320,  # Zeroscope 576w specific aspect ratio
        num_inference_steps=25,
        guidance_scale=9.0,
        num_frames=24,
        fps=8
    )
    
    print(f"Generating video with prompt: '{params.prompt}'")
    
    # Generate video
    result = generation_engine.generate_video(params)
    
    if result.error:
        print(f"❌ Video generation failed: {result.error}")
        return
    
    # Save the result
    saved_paths = result.save_images("outputs/examples", "zeroscope_video")
    
    # Save as video file
    try:
        video_path = generation_engine.save_result_as_video(
            result, "outputs/examples/zeroscope_video.mp4"
        )
        print(f"🎬 Video saved to: {video_path}")
    except Exception as e:
        print(f"⚠️  Failed to save video file: {e}")
    
    print(f"✅ Video generation completed in {result.generation_time:.2f}s")
    print(f"📁 Individual frames saved to: {saved_paths[0]}")
    print(f"🎲 Seed used: {result.seed_used}")

def main():
    """Run all video generation examples."""
    print("🎨 AI Generation Studio - Advanced Video Generation Examples")
    
    # Run basic video generation
    generate_basic_video()
    
    # Run ModelScope T2V generation
    generate_modelscope_video()
    
    # Run Zeroscope generation
    generate_zeroscope_video()
    
    print("\n" + "=" * 50)
    print("🎉 All video generation examples completed!")
    print("📁 Check the 'outputs/examples' directory for generated videos")
    print("=" * 50)

if __name__ == "__main__":
    main()