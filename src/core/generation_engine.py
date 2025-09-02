"""
Core generation engine for AI Generation Studio.
Handles image and video generation with various parameters and optimizations.
"""

import os
import logging
import uuid
import time
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from PIL import Image
import numpy as np

# Try to import torch, but handle gracefully if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

import sys
from pathlib import Path

# Add parent directory to path for imports
if __name__ != '__main__':
    current_dir = Path(__file__).resolve().parent
    src_dir = current_dir.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from .model_manager import get_model_manager, ModelType

try:
    from utils.gpu_utils import PerformanceMonitor, clear_gpu_cache
    from utils.config import get_config
except ImportError:
    # Fallback for relative imports
    try:
        from ..utils.gpu_utils import PerformanceMonitor, clear_gpu_cache
        from ..utils.config import get_config
    except ImportError:
        # Last resort - direct path import
        current_dir = Path(__file__).resolve().parent
        utils_dir = current_dir.parent / 'utils'
        sys.path.insert(0, str(utils_dir.parent))
        from utils.gpu_utils import PerformanceMonitor, clear_gpu_cache
        from utils.config import get_config

logger = logging.getLogger(__name__)

@dataclass
class GenerationParams:
    """Parameters for generation."""
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    seed: Optional[int] = None
    scheduler: str = "ddim"
    
    # Video-specific parameters
    num_frames: int = 16
    fps: int = 8
    
    # Advanced parameters
    clip_skip: int = 1
    eta: float = 0.0
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        self.width = max(64, min(2048, self.width))
        self.height = max(64, min(2048, self.height))
        self.num_inference_steps = max(1, min(100, self.num_inference_steps))
        self.guidance_scale = max(0.0, min(30.0, self.guidance_scale))
        self.num_images_per_prompt = max(1, min(10, self.num_images_per_prompt))

@dataclass
class GenerationResult:
    """Result of a generation operation."""
    images: List[Image.Image]
    metadata: Dict[str, Any]
    generation_time: float
    seed_used: int
    params: GenerationParams
    error: Optional[str] = None
    
    def save_images(self, output_dir: str, prefix: str = "generated") -> List[str]:
        """Save images to directory and return file paths."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        timestamp = int(time.time())
        
        for i, image in enumerate(self.images):
            filename = f"{prefix}_{timestamp}_{self.seed_used}_{i:03d}.png"
            filepath = output_path / filename
            
            # Add metadata to image
            metadata_str = f"Prompt: {self.params.prompt}\n"
            metadata_str += f"Seed: {self.seed_used}\n"
            metadata_str += f"Steps: {self.params.num_inference_steps}\n"
            metadata_str += f"Guidance: {self.params.guidance_scale}\n"
            metadata_str += f"Size: {self.params.width}x{self.params.height}"
            
            # Save with metadata
            image.save(filepath, pnginfo=self._create_png_info(metadata_str))
            saved_paths.append(str(filepath))
            
        return saved_paths
    
    def _create_png_info(self, metadata_str: str):
        """Create PNG info for metadata."""
        from PIL.PngImagePlugin import PngInfo
        pnginfo = PngInfo()
        pnginfo.add_text("parameters", metadata_str)
        return pnginfo

class GenerationEngine:
    """Core engine for AI generation."""
    
    def __init__(self):
        self.config = get_config()
        self.model_manager = get_model_manager()
        self.performance_monitor = PerformanceMonitor(self.model_manager.gpu_detector)
        self.generation_history = []
        
        # Callbacks for progress reporting
        self.progress_callback: Optional[Callable[[int, int, float], None]] = None
        self.status_callback: Optional[Callable[[str], None]] = None
        
        logger.info("GenerationEngine initialized")
    
    def set_progress_callback(self, callback: Callable[[int, int, float], None]):
        """Set callback for progress updates."""
        self.progress_callback = callback
    
    def set_status_callback(self, callback: Callable[[str], None]):
        """Set callback for status updates."""
        self.status_callback = callback
    
    def _update_status(self, status: str):
        """Update status through callback."""
        logger.info(status)
        if self.status_callback:
            self.status_callback(status)
    
    def _progress_callback_wrapper(self, pipe, step: int, timestep: float, latents: torch.FloatTensor):
        """Wrapper for diffusion pipeline progress callback."""
        if self.progress_callback:
            # Calculate progress percentage
            total_steps = pipe.num_timesteps if hasattr(pipe, 'num_timesteps') else 20
            progress = (step + 1) / total_steps
            self.progress_callback(step + 1, total_steps, progress)
    
    def generate_image(self, params: GenerationParams) -> GenerationResult:
        """Generate images from text prompt."""
        start_time = time.time()
        
        try:
            # Validate model is loaded
            if not self.model_manager.is_model_loaded():
                self._update_status("No model loaded. Loading default model...")
                if not self.model_manager.load_model("sd15"):
                    raise RuntimeError("Failed to load default model")
            
            # Check if current model supports the request
            current_model = self.model_manager.get_current_model()
            if current_model and current_model.supports_video and params.num_frames > 1:
                return self.generate_video(params)
            
            self._update_status("Preparing generation...")
            
            # Start performance monitoring
            self.performance_monitor.start_monitoring()
            
            # Get pipeline
            pipeline = self.model_manager.current_pipeline
            
            # Set up generator for reproducible results
            generator = None
            seed_used = params.seed
            if seed_used is None:
                if TORCH_AVAILABLE:
                    seed_used = torch.randint(0, 2**32 - 1, (1,)).item()
                else:
                    import random
                    seed_used = random.randint(0, 2**32 - 1)
            
            if TORCH_AVAILABLE:
                generator = torch.Generator(device=self.model_manager.device)
                generator.manual_seed(seed_used)
            
            self._update_status(f"Generating with seed: {seed_used}")
            
            # Prepare generation arguments
            gen_args = {
                "prompt": params.prompt,
                "negative_prompt": params.negative_prompt,
                "width": params.width,
                "height": params.height,
                "num_inference_steps": params.num_inference_steps,
                "guidance_scale": params.guidance_scale,
                "num_images_per_prompt": params.num_images_per_prompt,
                "generator": generator,
                "callback": self._progress_callback_wrapper,
                "callback_steps": 1
            }
            
            # Handle SDXL-specific parameters
            current_model = self.model_manager.get_current_model()
            if current_model and current_model.model_type == ModelType.STABLE_DIFFUSION_XL:
                # SDXL has different parameter names
                if hasattr(pipeline, 'refiner'):
                    gen_args["denoising_end"] = 0.8
            
            # Generate images
            self._update_status("Generating images...")
            if TORCH_AVAILABLE:
                with torch.inference_mode():
                    if pipeline is None:
                        raise RuntimeError("Pipeline not available")
                    result = pipeline(**gen_args)
            else:
                raise RuntimeError("PyTorch not available. Cannot generate images.")
            
            # Stop monitoring
            perf_stats = self.performance_monitor.stop_monitoring()
            
            generation_time = time.time() - start_time
            self._update_status(f"Generation completed in {generation_time:.2f}s")
            
            # Create result
            current_model = self.model_manager.get_current_model()
            generation_result = GenerationResult(
                images=result.images,
                metadata={
                    "performance": perf_stats,
                    "model": current_model.name if current_model else "Unknown",
                    "device": self.model_manager.device
                },
                generation_time=generation_time,
                seed_used=seed_used,
                params=params
            )
            
            # Add to history
            self.generation_history.append(generation_result)
            
            return generation_result
            
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            logger.error(error_msg)
            self._update_status(error_msg)
            
            return GenerationResult(
                images=[],
                metadata={},
                generation_time=time.time() - start_time,
                seed_used=params.seed or 0,
                params=params,
                error=error_msg
            )
    
    def generate_video(self, params: GenerationParams) -> GenerationResult:
        """Generate video from text prompt using AnimateDiff."""
        start_time = time.time()
        
        try:
            # Ensure we have a video-capable model
            current_model = self.model_manager.get_current_model()
            if not current_model or not current_model.supports_video:
                self._update_status("Loading video generation model...")
                if not self.model_manager.load_model("animatediff"):
                    raise RuntimeError("Failed to load video generation model")
            
            self._update_status("Preparing video generation...")
            
            # Start performance monitoring
            self.performance_monitor.start_monitoring()
            
            # Get pipeline
            pipeline = self.model_manager.current_pipeline
            
            # Set up generator
            generator = None
            seed_used = params.seed
            if seed_used is None:
                if TORCH_AVAILABLE:
                    seed_used = torch.randint(0, 2**32 - 1, (1,)).item()
                else:
                    import random
                    seed_used = random.randint(0, 2**32 - 1)
            
            if TORCH_AVAILABLE:
                generator = torch.Generator(device=self.model_manager.device)
                generator.manual_seed(seed_used)
            
            self._update_status(f"Generating video with seed: {seed_used}")
            
            # Prepare generation arguments for video
            gen_args = {
                "prompt": params.prompt,
                "negative_prompt": params.negative_prompt,
                "width": params.width,
                "height": params.height,
                "num_frames": params.num_frames,
                "num_inference_steps": params.num_inference_steps,
                "guidance_scale": params.guidance_scale,
                "generator": generator,
                "callback": self._progress_callback_wrapper,
                "callback_steps": 1
            }
            
            # Generate video
            self._update_status("Generating video frames...")
            if TORCH_AVAILABLE:
                with torch.inference_mode():
                    if pipeline is None:
                        raise RuntimeError("Pipeline not available")
                    result = pipeline(**gen_args)
            else:
                raise RuntimeError("PyTorch not available. Cannot generate videos.")
            
            # Convert video result to images if needed
            if hasattr(result, 'frames'):
                images = result.frames[0]  # First batch
            else:
                images = result.images
            
            # Stop monitoring
            perf_stats = self.performance_monitor.stop_monitoring()
            
            generation_time = time.time() - start_time
            self._update_status(f"Video generation completed in {generation_time:.2f}s")
            
            # Create result
            current_model = self.model_manager.get_current_model()
            generation_result = GenerationResult(
                images=images,
                metadata={
                    "performance": perf_stats,
                    "model": current_model.name if current_model else "Unknown",
                    "device": self.model_manager.device,
                    "video": True,
                    "num_frames": params.num_frames,
                    "fps": params.fps
                },
                generation_time=generation_time,
                seed_used=seed_used,
                params=params
            )
            
            # Add to history
            self.generation_history.append(generation_result)
            
            return generation_result
            
        except Exception as e:
            error_msg = f"Video generation failed: {str(e)}"
            logger.error(error_msg)
            self._update_status(error_msg)
            
            return GenerationResult(
                images=[],
                metadata={"video": True},
                generation_time=time.time() - start_time,
                seed_used=params.seed or 0,
                params=params,
                error=error_msg
            )
    
    def batch_generate(self, prompts: List[str], base_params: GenerationParams, 
                      progress_callback: Optional[Callable[[int, int], None]] = None) -> List[GenerationResult]:
        """Generate multiple images from a list of prompts."""
        results = []
        total_prompts = len(prompts)
        
        self._update_status(f"Starting batch generation for {total_prompts} prompts")
        
        for i, prompt in enumerate(prompts):
            # Update progress
            if progress_callback:
                progress_callback(i, total_prompts)
            
            # Create parameters for this prompt
            current_params = GenerationParams(
                prompt=prompt,
                negative_prompt=base_params.negative_prompt,
                width=base_params.width,
                height=base_params.height,
                num_inference_steps=base_params.num_inference_steps,
                guidance_scale=base_params.guidance_scale,
                num_images_per_prompt=base_params.num_images_per_prompt,
                seed=None,  # Generate random seed for each
                scheduler=base_params.scheduler
            )
            
            self._update_status(f"Generating image {i+1}/{total_prompts}: {prompt[:50]}...")
            
            # Generate
            result = self.generate_image(current_params)
            results.append(result)
            
            # Clear cache between generations to prevent memory buildup
            if i % 5 == 4:  # Every 5 generations
                clear_gpu_cache()
        
        if progress_callback:
            progress_callback(total_prompts, total_prompts)
        
        self._update_status(f"Batch generation completed. Generated {len(results)} images")
        return results
    
    def estimate_batch_time(self, prompts: List[str], params: GenerationParams) -> float:
        """Estimate total time for batch generation."""
        if not prompts:
            return 0.0
        
        # Estimate time for one image
        single_time = self.model_manager.estimate_generation_time(
            params.width, params.height, params.num_inference_steps
        )
        
        # Account for batch overhead
        total_images = len(prompts) * params.num_images_per_prompt
        overhead_factor = 1.1  # 10% overhead
        
        return single_time * total_images * overhead_factor
    
    def get_generation_history(self) -> List[GenerationResult]:
        """Get history of recent generations."""
        return self.generation_history[-50:]  # Last 50 generations
    
    def clear_history(self):
        """Clear generation history."""
        self.generation_history.clear()
        logger.info("Generation history cleared")
    
    def save_result_as_video(self, result: GenerationResult, output_path: str, fps: Optional[int] = None) -> str:
        """Save generation result as video file."""
        if not result.images:
            raise ValueError("No images to save as video")
        
        if fps is None:
            fps = result.params.fps if hasattr(result.params, 'fps') else 8
        
        try:
            import imageio
            
            # Convert PIL images to numpy arrays
            frames = []
            for img in result.images:
                frames.append(np.array(img))
            
            # Save as MP4
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            imageio.mimsave(str(output_path_obj), frames, fps=fps, quality=8)
            
            logger.info(f"Video saved to {output_path_obj}")
            return str(output_path_obj)
            
        except ImportError:
            raise RuntimeError("imageio required for video export. Install with: pip install imageio")
        except Exception as e:
            raise RuntimeError(f"Failed to save video: {e}")
    
    def interrupt_generation(self):
        """Interrupt current generation (if supported by model)."""
        # This is a placeholder - actual implementation would depend on
        # the specific pipeline and would require threading coordination
        self._update_status("Generation interruption requested")
        logger.info("Generation interruption requested")

# Global generation engine instance
_generation_engine = None

def get_generation_engine() -> GenerationEngine:
    """Get the global generation engine instance."""
    global _generation_engine
    if _generation_engine is None:
        _generation_engine = GenerationEngine()
    return _generation_engine