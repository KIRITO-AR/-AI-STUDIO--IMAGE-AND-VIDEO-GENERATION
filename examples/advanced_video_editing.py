"""
Advanced video editing example for AI Generation Studio.
This script demonstrates video editing capabilities including trimming, speed adjustment, and looping.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core import get_generation_engine, get_model_manager, GenerationParams

def generate_and_edit_video():
    """Generate a video and apply various editing operations."""
    print("🎬 AI Generation Studio - Advanced Video Editing Example")
    print("=" * 60)
    
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
        prompt="a red sports car driving through a futuristic city, cinematic, highly detailed",
        negative_prompt="blurry, low quality, static",
        width=512,
        height=512,
        num_inference_steps=20,
        guidance_scale=7.5,
        num_frames=24,
        fps=8
    )
    
    print(f"Generating video with prompt: '{params.prompt}'")
    
    # Generate video
    result = generation_engine.generate_video(params)
    
    if result.error:
        print(f"❌ Video generation failed: {result.error}")
        return
    
    print(f"✅ Video generation completed in {result.generation_time:.2f}s")
    print(f"📊 Generated {len(result.images)} frames")
    print(f"🎲 Seed used: {result.seed_used}")
    
    # Save original video
    try:
        original_video_path = generation_engine.save_result_as_video(
            result, "outputs/examples/original_video.mp4"
        )
        print(f"🎬 Original video saved to: {original_video_path}")
    except Exception as e:
        print(f"⚠️  Failed to save original video: {e}")
    
    # 1. Trim video - keep only the middle portion
    print("\n" + "-" * 40)
    print("✂️  Trimming video (frames 8-16)...")
    try:
        trimmed_result = generation_engine.edit_video(
            result, "trim", start_frame=8, end_frame=16
        )
        
        trimmed_video_path = generation_engine.save_result_as_video(
            trimmed_result, "outputs/examples/trimmed_video.mp4"
        )
        print(f"✅ Trimmed video saved to: {trimmed_video_path}")
        print(f"📊 Trimmed video has {len(trimmed_result.images)} frames")
    except Exception as e:
        print(f"❌ Failed to trim video: {e}")
    
    # 2. Adjust speed - make it faster
    print("\n" + "-" * 40)
    print("⚡ Speeding up video (2x speed)...")
    try:
        fast_result = generation_engine.edit_video(
            result, "speed", speed_factor=2.0
        )
        
        fast_video_path = generation_engine.save_result_as_video(
            fast_result, "outputs/examples/fast_video.mp4"
        )
        print(f"✅ Fast video saved to: {fast_video_path}")
        print(f"📊 Fast video has {len(fast_result.images)} frames")
    except Exception as e:
        print(f"❌ Failed to speed up video: {e}")
    
    # 3. Reverse video
    print("\n" + "-" * 40)
    print("🔄 Reversing video...")
    try:
        reversed_result = generation_engine.edit_video(
            result, "reverse"
        )
        
        reversed_video_path = generation_engine.save_result_as_video(
            reversed_result, "outputs/examples/reversed_video.mp4"
        )
        print(f"✅ Reversed video saved to: {reversed_video_path}")
        print(f"📊 Reversed video has {len(reversed_result.images)} frames")
    except Exception as e:
        print(f"❌ Failed to reverse video: {e}")
    
    # 4. Loop video
    print("\n" + "-" * 40)
    print("🔁 Creating looped video (2 loops)...")
    try:
        looped_result = generation_engine.edit_video(
            result, "loop", loops=2
        )
        
        looped_video_path = generation_engine.save_result_as_video(
            looped_result, "outputs/examples/looped_video.mp4"
        )
        print(f"✅ Looped video saved to: {looped_video_path}")
        print(f"📊 Looped video has {len(looped_result.images)} frames")
    except Exception as e:
        print(f"❌ Failed to loop video: {e}")
    
    # 5. Upscale video
    print("\n" + "-" * 40)
    print("⬆️  Upscaling video...")
    try:
        upscaled_video_path = generation_engine.upscale_video(
            result, "outputs/examples/upscaled_video.mp4"
        )
        print(f"✅ Upscaled video saved to: {upscaled_video_path}")
    except Exception as e:
        print(f"❌ Failed to upscale video: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 All video editing examples completed!")
    print("📁 Check the 'outputs/examples' directory for all generated videos")
    print("=" * 60)

def combine_videos_example():
    """Example of combining multiple videos with transitions."""
    print("\n" + "=" * 60)
    print("🎭 Video Combination Example")
    print("=" * 60)
    
    # This would be implemented using the VideoEditor class
    print("💡 This example shows how to combine multiple videos:")
    print("   1. Generate multiple video clips")
    print("   2. Add transitions between clips")
    print("   3. Concatenate into a final video")
    
    print("\n📝 Implementation would use:")
    print("   from src.models.video_generation import VideoEditor, VideoTransition")
    print("   combined_frames = VideoEditor.add_transition(frames1, frames2, VideoTransition.FADE)")
    print("   final_frames = VideoEditor.concatenate_videos([clip1, transition, clip2])")

def main():
    """Run all video editing examples."""
    generate_and_edit_video()
    combine_videos_example()

if __name__ == "__main__":
    main()