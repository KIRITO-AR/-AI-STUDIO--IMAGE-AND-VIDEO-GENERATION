"""
Video generation models and utilities for AI Generation Studio.
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from diffusers import (
        TextToVideoSDPipeline,
        VideoToVideoSDPipeline,
        DPMSolverMultistepScheduler
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    TextToVideoSDPipeline = None
    VideoToVideoSDPipeline = None
    DPMSolverMultistepScheduler = None

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class VideoTransition(Enum):
    """Types of video transitions."""
    CUT = "cut"
    FADE = "fade"
    DISSOLVE = "dissolve"
    SLIDE = "slide"

@dataclass
class VideoClip:
    """Represents a video clip with metadata."""
    frames: List[Image.Image]
    start_time: float = 0.0
    duration: float = 0.0
    fps: int = 8

class VideoEditor:
    """Basic video editing utilities."""
    
    @staticmethod
    def concatenate_videos(videos: List[List[Image.Image]], fps: int = 8) -> List[Image.Image]:
        """Concatenate multiple videos into one."""
        combined_frames = []
        for video_frames in videos:
            combined_frames.extend(video_frames)
        return combined_frames
    
    @staticmethod
    def trim_video(frames: List[Image.Image], start_frame: int, end_frame: int) -> List[Image.Image]:
        """Trim video to specified frame range."""
        if start_frame < 0:
            start_frame = 0
        if end_frame > len(frames):
            end_frame = len(frames)
        return frames[start_frame:end_frame]
    
    @staticmethod
    def add_transition(frames1: List[Image.Image], frames2: List[Image.Image], 
                      transition_type: VideoTransition = VideoTransition.CUT,
                      transition_frames: int = 4) -> List[Image.Image]:
        """Add transition between two video clips."""
        if transition_type == VideoTransition.CUT:
            # Simple cut - just concatenate
            return frames1 + frames2
        
        elif transition_type == VideoTransition.FADE:
            # Fade out first clip, fade in second clip
            result = frames1[:-transition_frames]  # Remove last few frames
            
            # Create fade transition frames
            for i in range(transition_frames):
                alpha = i / (transition_frames - 1)
                # Blend last frame of first clip with first frame of second clip
                blended = Image.blend(frames1[-1], frames2[0], alpha)
                result.append(blended)
            
            result.extend(frames2[1:])  # Add remaining frames
            return result
        
        elif transition_type == VideoTransition.DISSOLVE:
            # Dissolve transition
            result = frames1[:-transition_frames]
            
            # Create dissolve frames
            for i in range(transition_frames):
                alpha = i / (transition_frames - 1)
                idx1 = -(transition_frames - i)
                idx2 = i
                if idx1 < -len(frames1) or idx2 >= len(frames2):
                    continue
                blended = Image.blend(frames1[idx1], frames2[idx2], alpha)
                result.append(blended)
            
            result.extend(frames2[transition_frames:])
            return result
        
        else:  # Default to cut
            return frames1 + frames2
    
    @staticmethod
    def adjust_speed(frames: List[Image.Image], speed_factor: float) -> List[Image.Image]:
        """Adjust video playback speed."""
        if speed_factor <= 0:
            raise ValueError("Speed factor must be positive")
        
        if speed_factor == 1.0:
            return frames
        
        new_length = int(len(frames) / speed_factor)
        if new_length == 0:
            return [frames[0]] if frames else []
        
        # Simple frame sampling
        indices = np.linspace(0, len(frames) - 1, new_length).astype(int)
        return [frames[i] for i in indices]
    
    @staticmethod
    def reverse_video(frames: List[Image.Image]) -> List[Image.Image]:
        """Reverse video playback."""
        return frames[::-1]
    
    @staticmethod
    def loop_video(frames: List[Image.Image], loops: int) -> List[Image.Image]:
        """Loop video for specified number of times."""
        if loops <= 0:
            return []
        result = []
        for _ in range(loops):
            result.extend(frames)
        return result

class VideoGenerationModel:
    """Base class for video generation models."""
    
    def __init__(self, model_id: str, name: str):
        self.model_id = model_id
        self.name = name
        self.pipeline = None
        self.is_loaded = False
    
    def load(self, device: str = "cuda") -> bool:
        """Load the model pipeline."""
        if not DIFFUSERS_AVAILABLE:
            logger.error("Diffusers not available")
            return False
            
        try:
            logger.info(f"Loading video model: {self.name}")
            self.pipeline = TextToVideoSDPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if TORCH_AVAILABLE and torch.cuda.is_available() else torch.float32,
                variant="fp16" if TORCH_AVAILABLE and torch.cuda.is_available() else None
            )
            
            if TORCH_AVAILABLE and device == "cuda" and torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
            
            # Enable optimizations
            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
            
            if hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()
            
            self.is_loaded = True
            logger.info(f"Successfully loaded video model: {self.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load video model {self.name}: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> List[Image.Image]:
        """Generate video frames from prompt."""
        if not self.is_loaded or self.pipeline is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Prepare generation arguments
            gen_args = {
                "prompt": prompt,
                "num_inference_steps": kwargs.get("num_inference_steps", 25),
                "guidance_scale": kwargs.get("guidance_scale", 9.0),
                "num_frames": kwargs.get("num_frames", 16),
                "height": kwargs.get("height", 512),
                "width": kwargs.get("width", 512),
                "negative_prompt": kwargs.get("negative_prompt", "")
            }
            
            # Generate video frames
            if TORCH_AVAILABLE:
                with torch.inference_mode():
                    result = self.pipeline(**gen_args)
            else:
                result = self.pipeline(**gen_args)
            
            # Convert to PIL images
            if hasattr(result, 'frames'):
                frames = result.frames[0]  # First batch
            else:
                frames = result.images
            
            # Convert numpy arrays to PIL images if needed
            pil_frames = []
            for frame in frames:
                if isinstance(frame, np.ndarray):
                    # Convert numpy array to PIL Image
                    if frame.dtype == np.float32:
                        # Normalize float arrays to 0-255 range
                        frame = (frame * 255).astype(np.uint8)
                    pil_frames.append(Image.fromarray(frame))
                else:
                    pil_frames.append(frame)
            
            return pil_frames
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            raise

class ModelScopeT2V(VideoGenerationModel):
    """ModelScope Text-to-Video model."""
    
    def __init__(self):
        super().__init__("damo-vilab/text-to-video-ms-1.7b", "ModelScope T2V")
    
    def load(self, device: str = "cuda") -> bool:
        """Load the ModelScope T2V pipeline."""
        if not DIFFUSERS_AVAILABLE:
            return False
            
        try:
            logger.info("Loading ModelScope T2V model")
            self.pipeline = TextToVideoSDPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if TORCH_AVAILABLE and torch.cuda.is_available() else torch.float32,
                variant="fp16" if TORCH_AVAILABLE and torch.cuda.is_available() else None
            )
            
            if TORCH_AVAILABLE and device == "cuda" and torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
            
            # Enable optimizations
            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
            
            # Enable forward chunking for memory optimization
            if hasattr(self.pipeline.unet, 'enable_forward_chunking'):
                self.pipeline.unet.enable_forward_chunking(chunk_size=1, dim=1)
            
            if hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()
            
            self.is_loaded = True
            logger.info("Successfully loaded ModelScope T2V model")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ModelScope T2V model: {e}")
            return False

class ZeroscopeModel(VideoGenerationModel):
    """Zeroscope video generation model."""
    
    def __init__(self, model_variant: str = "576w"):
        if model_variant == "576w":
            model_id = "cerspense/zeroscope_v2_576w"
            name = "Zeroscope V2 576w"
        else:
            model_id = "cerspense/zeroscope_v2_XL"
            name = "Zeroscope V2 XL"
        
        super().__init__(model_id, name)
        self.model_variant = model_variant
    
    def load(self, device: str = "cuda") -> bool:
        """Load the Zeroscope pipeline."""
        if not DIFFUSERS_AVAILABLE:
            return False
            
        try:
            logger.info(f"Loading Zeroscope {self.model_variant} model")
            self.pipeline = TextToVideoSDPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if TORCH_AVAILABLE and torch.cuda.is_available() else torch.float32
            )
            
            if TORCH_AVAILABLE and device == "cuda" and torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
            
            # Enable optimizations
            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
            
            # Enable forward chunking for memory optimization
            if hasattr(self.pipeline.unet, 'enable_forward_chunking'):
                self.pipeline.unet.enable_forward_chunking(chunk_size=1, dim=1)
            
            if hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()
            
            self.is_loaded = True
            logger.info(f"Successfully loaded Zeroscope {self.model_variant} model")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Zeroscope {self.model_variant} model: {e}")
            return False

class VideoUpscaler:
    """Video upscaling using Zeroscope XL model."""
    
    def __init__(self):
        self.pipeline = None
        self.is_loaded = False
    
    def load(self, device: str = "cuda") -> bool:
        """Load the upscaling pipeline."""
        if not DIFFUSERS_AVAILABLE:
            return False
            
        try:
            logger.info("Loading Zeroscope XL upscaling model")
            self.pipeline = VideoToVideoSDPipeline.from_pretrained(
                "cerspense/zeroscope_v2_XL",
                torch_dtype=torch.float16 if TORCH_AVAILABLE and torch.cuda.is_available() else torch.float32
            )
            
            if TORCH_AVAILABLE and device == "cuda" and torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
            
            # Set scheduler
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )
            
            # Enable optimizations
            if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                self.pipeline.enable_model_cpu_offload()
            
            # Enable forward chunking for memory optimization
            if hasattr(self.pipeline.unet, 'enable_forward_chunking'):
                self.pipeline.unet.enable_forward_chunking(chunk_size=1, dim=1)
            
            if hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()
            
            self.is_loaded = True
            logger.info("Successfully loaded Zeroscope XL upscaling model")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Zeroscope XL upscaling model: {e}")
            return False
    
    def upscale_video(self, frames: List[Image.Image], prompt: str, **kwargs) -> List[Image.Image]:
        """Upscale video frames."""
        if not self.is_loaded or self.pipeline is None:
            raise RuntimeError("Upscaling model not loaded")
        
        try:
            # Resize frames for upscaling
            resized_frames = []
            target_size = (1024, 576)  # Zeroscope XL target size
            for frame in frames:
                resized_frame = frame.resize(target_size, Image.LANCZOS)
                resized_frames.append(resized_frame)
            
            # Prepare generation arguments
            gen_args = {
                "prompt": prompt,
                "video": resized_frames,
                "strength": kwargs.get("strength", 0.6),
                "num_inference_steps": kwargs.get("num_inference_steps", 25),
                "guidance_scale": kwargs.get("guidance_scale", 9.0),
                "negative_prompt": kwargs.get("negative_prompt", "")
            }
            
            # Generate upscaled video frames
            if TORCH_AVAILABLE:
                with torch.inference_mode():
                    result = self.pipeline(**gen_args)
            else:
                result = self.pipeline(**gen_args)
            
            # Convert to PIL images
            if hasattr(result, 'frames'):
                upscaled_frames = result.frames[0]  # First batch
            else:
                upscaled_frames = result.images
            
            # Convert numpy arrays to PIL images if needed
            pil_frames = []
            for frame in upscaled_frames:
                if isinstance(frame, np.ndarray):
                    # Convert numpy array to PIL Image
                    if frame.dtype == np.float32:
                        # Normalize float arrays to 0-255 range
                        frame = (frame * 255).astype(np.uint8)
                    pil_frames.append(Image.fromarray(frame))
                else:
                    pil_frames.append(frame)
            
            return pil_frames
            
        except Exception as e:
            logger.error(f"Video upscaling failed: {e}")
            raise

def get_video_model(model_name: str) -> Optional[VideoGenerationModel]:
    """Get a video generation model by name."""
    model_map = {
        "modelscope_t2v": ModelScopeT2V,
        "zeroscope_576w": lambda: ZeroscopeModel("576w"),
        "zeroscope_xl": lambda: ZeroscopeModel("xl")
    }
    
    if model_name in model_map:
        return model_map[model_name]()
    
    return None

def save_video(frames: List[Image.Image], output_path: str, fps: int = 8) -> str:
    """Save video frames as MP4 file."""
    try:
        import imageio
        
        # Convert PIL images to numpy arrays
        frame_arrays = []
        for img in frames:
            frame_arrays.append(np.array(img))
        
        # Save as MP4
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        imageio.mimsave(str(output_path_obj), frame_arrays, fps=fps, quality=8)
        
        logger.info(f"Video saved to {output_path_obj}")
        return str(output_path_obj)
        
    except ImportError:
        raise RuntimeError("imageio required for video export. Install with: pip install imageio")
    except Exception as e:
        raise RuntimeError(f"Failed to save video: {e}")