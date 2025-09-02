"""
Model management for AI Generation Studio.
Handles loading, caching, and switching between different AI models.
"""

import os
import logging
import gc
from typing import Dict, Optional, Any, Callable, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import threading
import time

# Try to import torch and diffusers, but handle gracefully if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        DiffusionPipeline,
        AutoencoderKL,
        DDIMScheduler,
        DDPMScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        DPMSolverMultistepScheduler
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    # Create dummy classes to prevent errors
    class DummyPipeline:
        pass
    StableDiffusionPipeline = DummyPipeline
    StableDiffusionXLPipeline = DummyPipeline
    DiffusionPipeline = DummyPipeline
    AutoencoderKL = DummyPipeline
    DDIMScheduler = DummyPipeline
    DDPMScheduler = DummyPipeline
    EulerAncestralDiscreteScheduler = DummyPipeline
    EulerDiscreteScheduler = DummyPipeline
    DPMSolverMultistepScheduler = DummyPipeline

try:
    from transformers import CLIPTextModel, CLIPTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    CLIPTextModel = None
    CLIPTokenizer = None

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
if __name__ != '__main__':
    current_dir = Path(__file__).resolve().parent
    src_dir = current_dir.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

try:
    from utils.gpu_utils import get_device_info, clear_gpu_cache, optimize_torch_settings
    from utils.config import get_config
except ImportError:
    # Fallback for relative imports
    try:
        from ..utils.gpu_utils import get_device_info, clear_gpu_cache, optimize_torch_settings
        from ..utils.config import get_config
    except ImportError:
        # Last resort - direct path import
        current_dir = Path(__file__).resolve().parent
        utils_dir = current_dir.parent / 'utils'
        sys.path.insert(0, str(utils_dir.parent))
        from utils.gpu_utils import get_device_info, clear_gpu_cache, optimize_torch_settings
        from utils.config import get_config

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model types."""
    STABLE_DIFFUSION_1_5 = "sd15"
    STABLE_DIFFUSION_XL = "sdxl" 
    ANIMATEDIFF = "animatediff"
    CUSTOM = "custom"

@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    model_type: ModelType
    model_id: str
    local_path: Optional[str] = None
    memory_requirement: int = 4000  # MB
    supports_video: bool = False
    supports_controlnet: bool = False
    description: str = ""

class ModelManager:
    """Manages AI models for generation."""
    
    def __init__(self):
        self.config = get_config()
        self.device, self.gpu_detector = get_device_info()
        self.current_pipeline = None
        self.current_model_info = None
        self.model_cache = {}
        self.lock = threading.Lock()
        
        # Check if required libraries are available
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available. Please install PyTorch to use AI models.")
        if not DIFFUSERS_AVAILABLE:
            logger.error("Diffusers not available. Please install diffusers to use AI models.")
        
        # Initialize torch optimizations if available
        if TORCH_AVAILABLE:
            optimize_torch_settings(self.gpu_detector)
        
        # Available models registry
        self.available_models = self._initialize_model_registry()
        
        logger.info(f"ModelManager initialized with device: {self.device}")
    
    def _initialize_model_registry(self) -> Dict[str, ModelInfo]:
        """Initialize the registry of available models."""
        return {
            "sd15": ModelInfo(
                name="Stable Diffusion 1.5",
                model_type=ModelType.STABLE_DIFFUSION_1_5,
                model_id="runwayml/stable-diffusion-v1-5",
                memory_requirement=4000,
                supports_video=False,
                supports_controlnet=True,
                description="The classic Stable Diffusion model, great for general purpose image generation"
            ),
            "sd15_inpainting": ModelInfo(
                name="Stable Diffusion 1.5 Inpainting",
                model_type=ModelType.STABLE_DIFFUSION_1_5,
                model_id="runwayml/stable-diffusion-inpainting",
                memory_requirement=4000,
                supports_video=False,
                supports_controlnet=True,
                description="Specialized for inpainting and editing existing images"
            ),
            "sdxl": ModelInfo(
                name="Stable Diffusion XL",
                model_type=ModelType.STABLE_DIFFUSION_XL,
                model_id="stabilityai/stable-diffusion-xl-base-1.0",
                memory_requirement=8000,
                supports_video=False,
                supports_controlnet=True,
                description="Higher resolution and improved quality over SD 1.5"
            ),
            "sdxl_turbo": ModelInfo(
                name="SDXL Turbo",
                model_type=ModelType.STABLE_DIFFUSION_XL,
                model_id="stabilityai/sdxl-turbo",
                memory_requirement=8000,
                supports_video=False,
                supports_controlnet=False,
                description="Fast generation with fewer steps, great for real-time applications"
            ),
            "animatediff": ModelInfo(
                name="AnimateDiff",
                model_type=ModelType.ANIMATEDIFF,
                model_id="guoyww/animatediff-motion-adapter-v1-5-2",
                memory_requirement=6000,
                supports_video=True,
                supports_controlnet=False,
                description="Generate short video clips from text prompts"
            )
        }
    
    def list_models(self) -> List[ModelInfo]:
        """List all available models."""
        return list(self.available_models.values())
    
    def get_model_info(self, model_key: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self.available_models.get(model_key)
    
    def check_model_compatibility(self, model_key: str) -> Dict[str, Any]:
        """Check if a model is compatible with current hardware."""
        model_info = self.get_model_info(model_key)
        if not model_info:
            return {"compatible": False, "reason": "Model not found"}
        
        best_gpu = self.gpu_detector.get_best_gpu()
        if not best_gpu:
            return {
                "compatible": True,
                "reason": "Will use CPU (slower performance)",
                "recommendations": ["Consider getting a GPU for better performance"]
            }
        
        available_memory = best_gpu.memory_free
        required_memory = model_info.memory_requirement
        
        if available_memory < required_memory:
            return {
                "compatible": False,
                "reason": f"Insufficient GPU memory. Required: {required_memory}MB, Available: {available_memory}MB",
                "recommendations": [
                    "Try closing other applications",
                    "Use CPU offloading",
                    "Choose a smaller model"
                ]
            }
        
        return {"compatible": True, "reason": "Fully compatible"}
    
    def load_model(self, model_key: str, **kwargs) -> bool:
        """Load a model for generation."""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available. Cannot load models.")
            return False
        
        if not DIFFUSERS_AVAILABLE:
            logger.error("Diffusers not available. Cannot load models.")
            return False
        
        with self.lock:
            try:
                model_info = self.get_model_info(model_key)
                if not model_info:
                    logger.error(f"Model {model_key} not found")
                    return False
                
                # Check compatibility
                compatibility = self.check_model_compatibility(model_key)
                if not compatibility["compatible"]:
                    logger.error(f"Model incompatible: {compatibility['reason']}")
                    return False
                
                # If model is already loaded, return early
                if (self.current_model_info and 
                    self.current_model_info.model_id == model_info.model_id):
                    logger.info(f"Model {model_key} already loaded")
                    return True
                
                # Clear previous model
                self._unload_current_model()
                
                # Load new model
                logger.info(f"Loading model: {model_info.name}")
                pipeline = self._create_pipeline(model_info, **kwargs)
                
                if pipeline is None:
                    logger.error(f"Failed to create pipeline for {model_key}")
                    return False
                
                # Apply optimizations
                self._apply_optimizations(pipeline, model_info)
                
                # Cache the pipeline
                self.current_pipeline = pipeline
                self.current_model_info = model_info
                self.model_cache[model_key] = pipeline
                
                logger.info(f"Successfully loaded model: {model_info.name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load model {model_key}: {e}")
                return False
    
    def _create_pipeline(self, model_info: ModelInfo, **kwargs) -> Optional[Any]:
        """Create a pipeline for the specified model."""
        try:
            # Get device and dtype
            device = self.device
            dtype = torch.float16 if device != "cpu" else torch.float32
            
            # Model-specific pipeline creation
            if model_info.model_type == ModelType.STABLE_DIFFUSION_XL:
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_info.model_id,
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    **kwargs
                )
            elif model_info.model_type == ModelType.ANIMATEDIFF:
                # AnimateDiff requires special handling
                pipeline = self._create_animatediff_pipeline(model_info, dtype, **kwargs)
            else:
                # Standard Stable Diffusion
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_info.model_id,
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    **kwargs
                )
            
            if pipeline is None:
                return None
            
            # Move to device
            pipeline = pipeline.to(device)
            
            # Set scheduler if specified
            if "scheduler" in kwargs:
                pipeline.scheduler = self._get_scheduler(kwargs["scheduler"], pipeline.scheduler.config)
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            return None
    
    def _create_animatediff_pipeline(self, model_info: ModelInfo, dtype, **kwargs):
        """Create AnimateDiff pipeline."""
        try:
            # This is a simplified version - actual AnimateDiff integration would be more complex
            try:
                from diffusers import AnimateDiffPipeline
            except ImportError:
                logger.error("AnimateDiff not available. Install with: pip install animatediff")
                return None
            
            pipeline = AnimateDiffPipeline.from_pretrained(
                "frankjoshua/toonyou_beta6",  # Base model
                motion_adapter=model_info.model_id,
                torch_dtype=dtype,
                **kwargs
            )
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to create AnimateDiff pipeline: {e}")
            return None
    
    def _get_scheduler(self, scheduler_name: str, config):
        """Get scheduler by name."""
        schedulers = {
            "ddim": DDIMScheduler,
            "ddpm": DDPMScheduler,
            "euler_a": EulerAncestralDiscreteScheduler,
            "euler": EulerDiscreteScheduler,
            "dpm": DPMSolverMultistepScheduler
        }
        
        scheduler_class = schedulers.get(scheduler_name.lower())
        if scheduler_class:
            return scheduler_class.from_config(config)
        
        logger.warning(f"Unknown scheduler: {scheduler_name}")
        return None
    
    def _apply_optimizations(self, pipeline, model_info: ModelInfo):
        """Apply optimizations to the pipeline."""
        try:
            # Get optimization settings
            opt_settings = self.gpu_detector.get_optimization_settings()
            memory_settings = self.gpu_detector.get_memory_recommendations()
            
            # Enable attention slicing for memory efficiency
            if memory_settings.get('use_attention_slicing', True):
                try:
                    pipeline.enable_attention_slicing()
                    logger.info("Attention slicing enabled")
                except:
                    pass
            
            # Enable CPU offloading if needed
            if memory_settings.get('enable_cpu_offload', False):
                try:
                    pipeline.enable_model_cpu_offload()
                    logger.info("Model CPU offloading enabled")
                except:
                    pass
            
            # Enable xformers if available
            if opt_settings.get('use_xformers', False):
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("xformers memory efficient attention enabled")
                except:
                    logger.warning("Failed to enable xformers")
            
            # Compile model if supported (PyTorch 2.0+)
            if opt_settings.get('use_torch_compile', False):
                try:
                    if hasattr(torch, 'compile'):
                        pipeline.unet = torch.compile(pipeline.unet)
                        logger.info("Model compiled with torch.compile")
                except:
                    logger.warning("Failed to compile model")
            
        except Exception as e:
            logger.warning(f"Failed to apply some optimizations: {e}")
    
    def _unload_current_model(self):
        """Unload the current model to free memory."""
        if self.current_pipeline is not None:
            logger.info("Unloading current model")
            
            # Move to CPU and delete
            try:
                if hasattr(self.current_pipeline, 'to'):
                    self.current_pipeline.to('cpu')
            except:
                pass
            
            del self.current_pipeline
            self.current_pipeline = None
            self.current_model_info = None
            
            # Force garbage collection and clear GPU cache
            gc.collect()
            clear_gpu_cache()
    
    def get_current_model(self) -> Optional[ModelInfo]:
        """Get currently loaded model info."""
        return self.current_model_info
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.current_pipeline is not None
    
    def estimate_generation_time(self, width: int, height: int, steps: int) -> float:
        """Estimate generation time based on parameters and hardware."""
        if not self.current_model_info:
            return 0.0
        
        # Base time estimates (in seconds) - these are rough estimates
        base_times = {
            ModelType.STABLE_DIFFUSION_1_5: 2.0,
            ModelType.STABLE_DIFFUSION_XL: 4.0,
            ModelType.ANIMATEDIFF: 10.0
        }
        
        base_time = base_times.get(self.current_model_info.model_type, 3.0)
        
        # Adjust for resolution
        pixels = width * height
        base_pixels = 512 * 512
        resolution_factor = pixels / base_pixels
        
        # Adjust for steps
        step_factor = steps / 20
        
        # Adjust for device
        device_factor = 1.0
        if self.device == "cpu":
            device_factor = 10.0  # CPU is much slower
        else:
            gpu = self.gpu_detector.get_best_gpu()
            if gpu and gpu.memory_total < 6000:  # Less than 6GB
                device_factor = 2.0
            elif gpu and gpu.memory_total < 12000:  # 6-12GB
                device_factor = 1.5
        
        estimated_time = base_time * resolution_factor * step_factor * device_factor
        return max(1.0, estimated_time)  # At least 1 second
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up ModelManager")
        self._unload_current_model()
        self.model_cache.clear()

# Global model manager instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

def cleanup_model_manager():
    """Clean up the global model manager."""
    global _model_manager
    if _model_manager is not None:
        _model_manager.cleanup()
        _model_manager = None