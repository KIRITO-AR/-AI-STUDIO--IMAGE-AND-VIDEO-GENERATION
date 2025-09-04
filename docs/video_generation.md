# Video Generation in AI Generation Studio

This document provides detailed information about video generation capabilities in AI Generation Studio.

## Overview

AI Generation Studio supports multiple video generation models, allowing users to create high-quality videos from text prompts. The application provides both basic and advanced video generation features with various quality and performance options.

## Supported Video Models

### 1. AnimateDiff
- **Model ID**: `guoyww/animatediff-motion-adapter-v1-5-2`
- **VRAM Requirements**: 6GB+
- **Description**: Basic video generation model, good for simple animations
- **Best For**: Character movements, simple scene transitions

### 2. ModelScope Text-to-Video
- **Model ID**: `damo-vilab/text-to-video-ms-1.7b`
- **VRAM Requirements**: 7GB+
- **Description**: Advanced text-to-video model with better quality and consistency
- **Best For**: Cinematic scenes, storytelling, complex animations

### 3. Zeroscope V2 576w
- **Model ID**: `cerspense/zeroscope_v2_576w`
- **VRAM Requirements**: 8GB+
- **Description**: High-quality, watermark-free video generation
- **Resolution**: 576x320
- **Best For**: High-quality short videos

### 4. Zeroscope V2 XL
- **Model ID**: `cerspense/zeroscope_v2_XL`
- **VRAM Requirements**: 12GB+
- **Description**: Video upscaling model for higher resolution output
- **Resolution**: 1024x576
- **Best For**: Upscaling lower resolution videos

## Video Generation Parameters

### Basic Parameters
- **Prompt**: Text description of the video content
- **Negative Prompt**: Elements to avoid in the video
- **Number of Frames**: Total frames in the video (8-64)
- **FPS**: Frames per second (4-30)
- **Width/Height**: Video resolution
- **Inference Steps**: Quality vs. speed tradeoff (10-50)
- **Guidance Scale**: Prompt adherence (1.0-20.0)

### Advanced Parameters
- **Seed**: For reproducible results
- **Scheduler**: Diffusion scheduler algorithm
- **Frame Interpolation**: Smooth frame transitions
- **Video Looping**: Create looping videos
- **Video Upscaling**: Enhance video resolution

## Usage Examples

### Basic Video Generation
```python
from core import get_generation_engine, get_model_manager, GenerationParams

# Initialize
model_manager = get_model_manager()
generation_engine = get_generation_engine()

# Load video model
model_manager.load_model("animatediff")

# Create parameters
params = GenerationParams(
    prompt="a beautiful sunset over mountains",
    num_frames=16,
    fps=8,
    width=512,
    height=512
)

# Generate video
result = generation_engine.generate_video(params)

# Save as video file
video_path = generation_engine.save_result_as_video(result, "output.mp4")
```

### Advanced Video Generation with ModelScope
```python
# Load advanced model
model_manager.load_model("modelscope_t2v")

# Create high-quality parameters
params = GenerationParams(
    prompt="cinematic scene of a futuristic city with flying cars",
    negative_prompt="blurry, low quality",
    num_frames=24,
    fps=8,
    width=512,
    height=512,
    num_inference_steps=25,
    guidance_scale=9.0
)

# Generate
result = generation_engine.generate_video(params)
```

### Zeroscope High-Quality Generation
```python
# Load Zeroscope model
model_manager.load_model("zeroscope_v2_576w")

# Zeroscope works best with specific resolution
params = GenerationParams(
    prompt="an astronaut riding a horse on mars",
    num_frames=24,
    fps=8,
    width=576,
    height=320  # Specific aspect ratio for Zeroscope
)

# Generate
result = generation_engine.generate_video(params)
```

## Video Editing Features

AI Generation Studio provides basic video editing capabilities:

### 1. Trimming
Remove unwanted portions from the beginning or end of a video:
```python
# Trim video to frames 5-20
trimmed_result = generation_engine.edit_video(
    result, "trim", start_frame=5, end_frame=20
)
```

### 2. Speed Adjustment
Change video playback speed:
```python
# Make video 2x faster
fast_result = generation_engine.edit_video(
    result, "speed", speed_factor=2.0
)

# Make video 0.5x slower (2x duration)
slow_result = generation_engine.edit_video(
    result, "speed", speed_factor=0.5
)
```

### 3. Reverse Playback
Play video in reverse:
```python
# Reverse video playback
reversed_result = generation_engine.edit_video(
    result, "reverse"
)
```

### 4. Looping
Repeat video multiple times:
```python
# Loop video 3 times
looped_result = generation_engine.edit_video(
    result, "loop", loops=3
)
```

## Video Upscaling

For higher resolution videos, you can upscale using the Zeroscope XL model:

1. Generate video with base model (576x320)
2. Upscale to 1024x576 using Zeroscope XL
3. Requires 12GB+ VRAM

```python
# Upscale video to higher resolution
upscaled_video_path = generation_engine.upscale_video(
    result, "upscaled_output.mp4"
)
```

## Performance Tips

### Memory Optimization
- Use `enable_model_cpu_offload()` for lower VRAM systems
- Enable `vae_slicing` to reduce memory usage
- Use `forward_chunking` for large frame counts

### Quality vs. Speed
- Higher inference steps = better quality but slower
- Lower FPS = smaller file size but choppier motion
- More frames = longer videos but larger files

### Hardware Recommendations
- **Minimum**: 6GB VRAM (AnimateDiff)
- **Recommended**: 8GB+ VRAM (Zeroscope)
- **High-End**: 12GB+ VRAM (Zeroscope XL upscaling)

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce frame count, resolution, or enable CPU offloading
2. **Poor Quality**: Increase inference steps or guidance scale
3. **Inconsistent Motion**: Try ModelScope T2V for better temporal consistency

### Error Handling
```python
try:
    result = generation_engine.generate_video(params)
    if result.error:
        print(f"Generation failed: {result.error}")
except Exception as e:
    print(f"Error: {e}")
```

## API Reference

### GenerationParams for Video
```python
class GenerationParams:
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    scheduler: str = "ddim"
    
    # Video-specific parameters
    num_frames: int = 16
    fps: int = 8
```

### GenerationResult for Video
```python
class GenerationResult:
    images: List[Image.Image]  # Video frames
    metadata: Dict[str, Any]
    generation_time: float
    seed_used: int
    params: GenerationParams
    error: Optional[str] = None
    
    def save_images(self, output_dir: str, prefix: str = "generated") -> List[str]:
        """Save video frames as images"""
    
    def save_as_video(self, output_path: str, fps: Optional[int] = None) -> str:
        """Save as video file"""
```

## Future Improvements

Planned enhancements:
- Audio generation integration
- Advanced video editing capabilities
- Multi-model video blending
- Advanced interpolation techniques
- Real-time video generation
- Video-to-video generation
- Video inpainting and outpainting