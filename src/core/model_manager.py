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

# Disable ONNX runtime to avoid DLL loading issues on Windows
import os
os.environ['DIFFUSERS_DISABLE_ONNX'] = '1'

try:
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline  # type: ignore
    from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline  # type: ignore
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL  # type: ignore
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler  # type: ignore
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # type: ignore
    from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler  # type: ignore
    from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler  # type: ignore
    from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler  # type: ignore
    # Import FLUX pipeline
    from diffusers import FluxPipeline  # type: ignore
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    DIFFUSERS_AVAILABLE = False
    # Create dummy classes to prevent errors
    class DummyPipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("Diffusers not available")
    
    StableDiffusionPipeline = DummyPipeline  # type: ignore
    StableDiffusionXLPipeline = DummyPipeline  # type: ignore
    DiffusionPipeline = DummyPipeline  # type: ignore
    AutoencoderKL = DummyPipeline  # type: ignore
    DDIMScheduler = DummyPipeline  # type: ignore
    DDPMScheduler = DummyPipeline  # type: ignore
    EulerAncestralDiscreteScheduler = DummyPipeline  # type: ignore
    EulerDiscreteScheduler = DummyPipeline  # type: ignore
    DPMSolverMultistepScheduler = DummyPipeline  # type: ignore
    FluxPipeline = DummyPipeline  # type: ignore

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
    from utils.gpu_utils import get_device_info, PerformanceMonitor
    from utils.config import get_config
except ImportError:
    # Fallback for relative imports
    try:
        from ..utils.gpu_utils import get_device_info, PerformanceMonitor
        from ..utils.config import get_config
    except ImportError:
        # Last resort - direct path import
        current_dir = Path(__file__).resolve().parent
        utils_dir = current_dir.parent / 'utils'
        sys.path.insert(0, str(utils_dir.parent))
        from utils.gpu_utils import get_device_info, PerformanceMonitor
        from utils.config import get_config

def clear_gpu_cache():
    """Clear GPU cache if available."""
    if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def optimize_torch_settings(gpu_detector):
    """Optimize PyTorch settings based on GPU capabilities."""
    if TORCH_AVAILABLE and torch is not None:
        # Enable optimizations if available
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model types."""
    STABLE_DIFFUSION_1_5 = "sd15"
    STABLE_DIFFUSION_XL = "sdxl" 
    ANIMATEDIFF = "animatediff"
    FLUX = "flux"
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
                description="Generate short video clips from text prompts (requires 6GB+ VRAM)"
            ),
            # Advanced video models
            "modelscope_t2v": ModelInfo(
                name="ModelScope Text-to-Video",
                model_type=ModelType.ANIMATEDIFF,  # Using same type for now
                model_id="damo-vilab/text-to-video-ms-1.7b",
                memory_requirement=7000,
                supports_video=True,
                supports_controlnet=False,
                description="Advanced text-to-video model with better quality and consistency (requires 7GB+ VRAM)"
            ),
            "zeroscope_v2_576w": ModelInfo(
                name="Zeroscope V2 576w",
                model_type=ModelType.ANIMATEDIFF,  # Using same type for now
                model_id="cerspense/zeroscope_v2_576w",
                memory_requirement=8000,
                supports_video=True,
                supports_controlnet=False,
                description="High-quality video generation model with watermark-free output (requires 8GB+ VRAM)"
            ),
            "zeroscope_v2_XL": ModelInfo(
                name="Zeroscope V2 XL",
                model_type=ModelType.ANIMATEDIFF,  # Using same type for now
                model_id="cerspense/zeroscope_v2_XL",
                memory_requirement=12000,
                supports_video=True,
                supports_controlnet=False,
                description="High-resolution video upscaling model (requires 12GB+ VRAM)"
            ),
            # Advanced models for cloud GPUs with high VRAM
            "flux_dev": ModelInfo(
                name="FLUX.1-dev",
                model_type=ModelType.FLUX,
                model_id="black-forest-labs/FLUX.1-dev",
                memory_requirement=24000,
                supports_video=False,
                supports_controlnet=False,
                description="State-of-the-art image generation with exceptional quality (requires 24GB+ VRAM)"
            ),
            "flux_schnell": ModelInfo(
                name="FLUX.1-schnell",
                model_type=ModelType.FLUX,
                model_id="black-forest-labs/FLUX.1-schnell",
                memory_requirement=16000,
                supports_video=False,
                supports_controlnet=False,
                description="Fast version of FLUX with excellent quality (requires 16GB+ VRAM)"
            ),
            # New powerful models
            "sd3_medium": ModelInfo(
                name="Stable Diffusion 3 Medium",
                model_type=ModelType.STABLE_DIFFUSION_XL,
                model_id="stabilityai/stable-diffusion-3-medium",
                memory_requirement=12000,
                supports_video=False,
                supports_controlnet=True,
                description="Latest Stable Diffusion 3 model with improved quality and efficiency (requires 12GB+ VRAM)"
            ),
            "sd3_5_large": ModelInfo(
                name="Stable Diffusion 3.5 Large",
                model_type=ModelType.STABLE_DIFFUSION_XL,
                model_id="stabilityai/stable-diffusion-3.5-large",
                memory_requirement=24000,
                supports_video=False,
                supports_controlnet=True,
                description="Most powerful Stable Diffusion 3.5 model with exceptional quality (requires 24GB+ VRAM)"
            ),
            "hunyuan_dit": ModelInfo(
                name="Hunyuan DiT",
                model_type=ModelType.CUSTOM,
                model_id="Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers",
                memory_requirement=24000,
                supports_video=False,
                supports_controlnet=False,
                description="Tencent's state-of-the-art DiT architecture model with multilingual support (requires 24GB+ VRAM)"
            ),
            "pixart_alpha": ModelInfo(
                name="PixArt-Alpha",
                model_type=ModelType.CUSTOM,
                model_id="PixArt-alpha/PixArt-XL-2-1024-MS",
                memory_requirement=12000,
                supports_video=False,
                supports_controlnet=False,
                description="Highly efficient transformer-based model with excellent image quality (requires 12GB+ VRAM)"
            ),
            "kolors": ModelInfo(
                name="Kolors",
                model_type=ModelType.CUSTOM,
                model_id="Kwai-Kolors/Kolors-diffusers",
                memory_requirement=24000,
                supports_video=False,
                supports_controlnet=False,
                description="Bytedance's advanced text-to-image model with multilingual support (requires 24GB+ VRAM)"
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
        total_gpu_memory = best_gpu.memory_total
        
        # Cloud GPU optimization (48GB+)
        if total_gpu_memory >= 48000:  # 48GB+ (cloud GPUs)
            return {
                "compatible": True,
                "reason": f"Cloud GPU detected ({int(total_gpu_memory/1024)}GB) - fully compatible with all models",
                "recommendations": [
                    "Optimal performance expected",
                    "Enable batch processing for faster generation",
                    "Use maximum resolution settings",
                    "Consider running multiple models simultaneously"
                ]
            }
        
        # Special handling for 4GB GPUs like GTX 1650
        if total_gpu_memory <= 4096:  # 4GB or less
            # For 4GB GPUs, be more flexible with memory requirements
            if model_info.model_type == ModelType.STABLE_DIFFUSION_1_5:
                # SD 1.5 models can work with aggressive optimizations on 4GB GPUs
                if available_memory >= 2500:  # At least 2.5GB free
                    return {
                        "compatible": True, 
                        "reason": "Compatible with memory optimizations",
                        "recommendations": [
                            "Using CPU offloading for memory efficiency",
                            "Attention slicing enabled",
                            "VAE slicing enabled"
                        ]
                    }
            elif model_info.model_type == ModelType.STABLE_DIFFUSION_XL:
                # SDXL is too heavy for 4GB GPUs, suggest alternatives
                return {
                    "compatible": False,
                    "reason": f"SDXL requires more memory than available on 4GB GPU. Required: {required_memory}MB, Available: {available_memory}MB",
                    "recommendations": [
                        "Use Stable Diffusion 1.5 instead",
                        "Consider upgrading to a GPU with 8GB+ VRAM",
                        "Use CPU mode (very slow)"
                    ]
                }
            elif model_info.model_type == ModelType.CUSTOM:
                # CUSTOM models like Bytedance Kolors, PixArt, etc. need more memory
                return {
                    "compatible": False,
                    "reason": f"Advanced models require more memory. Required: {required_memory}MB, Available: {available_memory}MB",
                    "recommendations": [
                        "Use Stable Diffusion 1.5 instead",
                        "Consider cloud GPU instances with 24GB+ VRAM",
                        "Upgrade to RTX 4090 or similar high-end GPU"
                    ]
                }
            elif model_info.model_type == ModelType.FLUX:
                # FLUX models need significant memory
                return {
                    "compatible": False,
                    "reason": f"FLUX models require more memory. Required: {required_memory}MB, Available: {available_memory}MB",
                    "recommendations": [
                        "Use Stable Diffusion 1.5 instead",
                        "Consider cloud GPU instances with 16GB+ VRAM",
                        "Upgrade to RTX 4080/4090 or similar"
                    ]
                }
        
        # Standard compatibility check for other GPUs
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
                
                # Load new model based on model type
                if model_info.model_type == ModelType.ANIMATEDIFF:
                    success = self._load_animatediff_model(model_info, **kwargs)
                elif model_info.model_type == ModelType.STABLE_DIFFUSION_1_5:
                    success = self._load_stable_diffusion_model(model_info, **kwargs)
                elif model_info.model_type == ModelType.STABLE_DIFFUSION_XL:
                    success = self._load_sdxl_model(model_info, **kwargs)
                elif model_info.model_type == ModelType.FLUX:
                    success = self._load_flux_model(model_info, **kwargs)
                elif model_info.model_type == ModelType.CUSTOM:
                    success = self._load_custom_model(model_info, **kwargs)
                else:
                    logger.error(f"Unsupported model type: {model_info.model_type}")
                    return False
                
                if success:
                    self.current_model_info = model_info
                    logger.info(f"Successfully loaded model: {model_info.name}")
                else:
                    logger.error(f"Failed to load model: {model_info.name}")
                
                return success
                
            except Exception as e:
                logger.error(f"Error loading model {model_key}: {e}")
                return False
    
    def _load_animatediff_model(self, model_info: ModelInfo, **kwargs) -> bool:
        """Load AnimateDiff model."""
        try:
            # Special handling for different video models
            if model_info.model_id == "damo-vilab/text-to-video-ms-1.7b":
                # ModelScope T2V
                from diffusers import TextToVideoSDPipeline
                pipeline_cls = TextToVideoSDPipeline
            elif "zeroscope" in model_info.model_id:
                # Zeroscope models
                from diffusers import TextToVideoSDPipeline
                pipeline_cls = TextToVideoSDPipeline
            else:
                # Default AnimateDiff
                from diffusers import DiffusionPipeline
                pipeline_cls = DiffusionPipeline
            
            logger.info(f"Loading AnimateDiff model: {model_info.name}")
            
            # Load pipeline
            if model_info.local_path:
                self.current_pipeline = pipeline_cls.from_pretrained(
                    model_info.local_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    **kwargs
                )
            else:
                self.current_pipeline = pipeline_cls.from_pretrained(
                    model_info.model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    **kwargs
                )
            
            # Move to device
            if torch.cuda.is_available():
                self.current_pipeline = self.current_pipeline.to(self.device)
            
            # Enable optimizations
            if hasattr(self.current_pipeline, 'enable_model_cpu_offload'):
                self.current_pipeline.enable_model_cpu_offload()
            
            # Model-specific optimizations
            if model_info.model_id == "damo-vilab/text-to-video-ms-1.7b":
                # ModelScope T2V specific optimizations
                if hasattr(self.current_pipeline.unet, 'enable_forward_chunking'):
                    self.current_pipeline.unet.enable_forward_chunking(chunk_size=1, dim=1)
                if hasattr(self.current_pipeline, 'enable_vae_slicing'):
                    self.current_pipeline.enable_vae_slicing()
            elif "zeroscope" in model_info.model_id:
                # Zeroscope specific optimizations
                if hasattr(self.current_pipeline.unet, 'enable_forward_chunking'):
                    self.current_pipeline.unet.enable_forward_chunking(chunk_size=1, dim=1)
                if hasattr(self.current_pipeline, 'enable_vae_slicing'):
                    self.current_pipeline.enable_vae_slicing()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load AnimateDiff model {model_info.name}: {e}")
            return False
    
    def _load_stable_diffusion_model(self, model_info: ModelInfo, **kwargs) -> bool:
        """Load Stable Diffusion 1.5 model."""
        try:
            logger.info(f"Loading Stable Diffusion 1.5 model: {model_info.name}")
            
            # Load pipeline
            if model_info.local_path:
                self.current_pipeline = StableDiffusionPipeline.from_pretrained(
                    model_info.local_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    **kwargs
                )
            else:
                self.current_pipeline = StableDiffusionPipeline.from_pretrained(
                    model_info.model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    **kwargs
                )
            
            # Move to device
            if torch.cuda.is_available():
                self.current_pipeline = self.current_pipeline.to(self.device)
            
            # Enable optimizations
            if hasattr(self.current_pipeline, 'enable_model_cpu_offload'):
                self.current_pipeline.enable_model_cpu_offload()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion 1.5 model {model_info.name}: {e}")
            return False
    
    def _load_sdxl_model(self, model_info: ModelInfo, **kwargs) -> bool:
        """Load Stable Diffusion XL model."""
        try:
            logger.info(f"Loading Stable Diffusion XL model: {model_info.name}")
            
            # Load pipeline
            if model_info.local_path:
                self.current_pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_info.local_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    **kwargs
                )
            else:
                self.current_pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_info.model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    **kwargs
                )
            
            # Move to device
            if torch.cuda.is_available():
                self.current_pipeline = self.current_pipeline.to(self.device)
            
            # Enable optimizations
            if hasattr(self.current_pipeline, 'enable_model_cpu_offload'):
                self.current_pipeline.enable_model_cpu_offload()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion XL model {model_info.name}: {e}")
            return False
    
    def _load_flux_model(self, model_info: ModelInfo, **kwargs) -> bool:
        """Load FLUX model."""
        try:
            logger.info(f"Loading FLUX model: {model_info.name}")
            
            # Load pipeline
            if model_info.local_path:
                self.current_pipeline = FluxPipeline.from_pretrained(
                    model_info.local_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    **kwargs
                )
            else:
                self.current_pipeline = FluxPipeline.from_pretrained(
                    model_info.model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    **kwargs
                )
            
            # Move to device
            if torch.cuda.is_available():
                self.current_pipeline = self.current_pipeline.to(self.device)
            
            # Enable optimizations
            if hasattr(self.current_pipeline, 'enable_model_cpu_offload'):
                self.current_pipeline.enable_model_cpu_offload()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load FLUX model {model_info.name}: {e}")
            return False
    
    def _load_custom_model(self, model_info: ModelInfo, **kwargs) -> bool:
        """Load custom model."""
        try:
            logger.info(f"Loading custom model: {model_info.name}")
            
            # Load pipeline
            if model_info.local_path:
                self.current_pipeline = DiffusionPipeline.from_pretrained(
                    model_info.local_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    **kwargs
                )
            else:
                self.current_pipeline = DiffusionPipeline.from_pretrained(
                    model_info.model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    **kwargs
                )
            
            # Move to device
            if torch.cuda.is_available():
                self.current_pipeline = self.current_pipeline.to(self.device)
            
            # Enable optimizations
            if hasattr(self.current_pipeline, 'enable_model_cpu_offload'):
                self.current_pipeline.enable_model_cpu_offload()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load custom model {model_info.name}: {e}")
            return False
    
    def _create_pipeline(self, model_info: ModelInfo, **kwargs) -> Optional[Any]:
        """Create a pipeline for the specified model."""
        try:
            # Get device and dtype
            device = self.device
            if TORCH_AVAILABLE and torch is not None:
                dtype = torch.float16 if device != "cpu" else torch.float32
            else:
                raise RuntimeError("PyTorch not available")
            
            # Model-specific pipeline creation
            if not DIFFUSERS_AVAILABLE:
                raise RuntimeError("Diffusers not available")
            
            # Set environment variable to disable ONNX to avoid DLL issues
            os.environ['DISABLE_ONNX'] = '1'
            
            # Add extra kwargs to prevent ONNX usage
            kwargs.update({
                'use_onnx': False,
                'provider': None  # Disable ONNX provider
            })
                
            if model_info.model_type == ModelType.STABLE_DIFFUSION_XL:
                # Use the imported class directly - type checker knows it's real when DIFFUSERS_AVAILABLE is True
                pipeline = StableDiffusionXLPipeline.from_pretrained(  # type: ignore
                    model_info.model_id,
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    **kwargs
                )
            elif model_info.model_type == ModelType.ANIMATEDIFF:
                # AnimateDiff requires special handling
                pipeline = self._create_animatediff_pipeline(model_info, dtype, **kwargs)
            elif model_info.model_type == ModelType.FLUX:
                # FLUX models use FluxPipeline
                pipeline = self._create_flux_pipeline(model_info, dtype, **kwargs)
            elif model_info.model_type == ModelType.CUSTOM:
                # CUSTOM models like Bytedance Kolors, PixArt, etc. use DiffusionPipeline
                pipeline = self._create_custom_pipeline(model_info, dtype, **kwargs)
            else:
                # Standard Stable Diffusion
                pipeline = StableDiffusionPipeline.from_pretrained(  # type: ignore
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
            # Check if this is an ONNX-related error
            error_str = str(e).lower()
            if 'onnx' in error_str or 'dll' in error_str:
                logger.error(f"ONNX runtime error detected. This is usually due to missing Visual C++ redistributables on Windows.")
                logger.error(f"To fix this issue, either:")
                logger.error(f"1. Install Microsoft Visual C++ Redistributable (recommended)")
                logger.error(f"2. Or uninstall onnxruntime: pip uninstall onnxruntime onnxruntime-gpu")
                logger.error(f"Original error: {e}")
            else:
                logger.error(f"Failed to create pipeline: {e}")
            return None
    
    def _create_animatediff_pipeline(self, model_info: ModelInfo, dtype, **kwargs):
        """Create AnimateDiff pipeline."""
        try:
            # This is a simplified version - actual AnimateDiff integration would be more complex
            try:
                from diffusers.pipelines.animatediff.pipeline_animatediff import AnimateDiffPipeline  # type: ignore
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
    
    def _create_flux_pipeline(self, model_info: ModelInfo, dtype, **kwargs):
        """Create FLUX pipeline."""
        try:
            logger.info(f"Creating FLUX pipeline for {model_info.name}")
            
            # Remove conflicting kwargs for FLUX
            flux_kwargs = kwargs.copy()
            flux_kwargs.pop('safety_checker', None)
            flux_kwargs.pop('requires_safety_checker', None)
            flux_kwargs.pop('use_onnx', None)
            flux_kwargs.pop('provider', None)
            
            # Use bfloat16 for FLUX as recommended
            if TORCH_AVAILABLE and torch is not None and hasattr(torch, 'bfloat16'):
                flux_dtype = torch.bfloat16
            else:
                flux_dtype = dtype
            
            # Create FLUX pipeline
            pipeline = FluxPipeline.from_pretrained(  # type: ignore
                model_info.model_id,
                torch_dtype=flux_dtype,
                **flux_kwargs
            )
            
            logger.info(f"FLUX pipeline created successfully")
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to create FLUX pipeline: {e}")
            # If this is a 401 error, provide specific guidance
            if '401' in str(e) or 'authorization' in str(e).lower():
                logger.error("FLUX model access is restricted. You need to:")
                logger.error("1. Create a Hugging Face account at https://huggingface.co")
                logger.error("2. Request access to the FLUX model repository")
                logger.error("3. Install and configure huggingface-hub: pip install huggingface-hub")
                logger.error("4. Login with: huggingface-cli login")
            return None
    
    def _create_custom_pipeline(self, model_info: ModelInfo, dtype, **kwargs):
        """Create CUSTOM pipeline for models like Bytedance Kolors, PixArt, etc.."""
        try:
            logger.info(f"Creating CUSTOM pipeline for {model_info.name}")
            
            # Remove conflicting kwargs for CUSTOM models
            custom_kwargs = kwargs.copy()
            custom_kwargs.pop('safety_checker', None)
            custom_kwargs.pop('requires_safety_checker', None)
            custom_kwargs.pop('use_onnx', None)
            custom_kwargs.pop('provider', None)
            
            # Use appropriate dtype for CUSTOM models
            if TORCH_AVAILABLE and torch is not None and hasattr(torch, 'bfloat16'):
                custom_dtype = torch.bfloat16
            else:
                custom_dtype = dtype
            
            # Create CUSTOM pipeline using DiffusionPipeline
            from diffusers import DiffusionPipeline
            pipeline = DiffusionPipeline.from_pretrained(
                model_info.model_id,
                torch_dtype=custom_dtype,
                **custom_kwargs
            )
            
            logger.info(f"CUSTOM pipeline created successfully for {model_info.name}")
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to create CUSTOM pipeline for {model_info.name}: {e}")
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
            
            # FLUX models need specific optimizations
            if model_info.model_type == ModelType.FLUX:
                # Enable CPU offloading for FLUX (recommended for memory efficiency)
                try:
                    pipeline.enable_model_cpu_offload()
                    logger.info("FLUX: Model CPU offloading enabled")
                except Exception as e:
                    logger.warning(f"Failed to enable CPU offloading for FLUX: {e}")
                
                # Don't apply some optimizations that might conflict with FLUX
                logger.info("FLUX optimizations applied")
                return
            
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
                    if TORCH_AVAILABLE and torch is not None and hasattr(torch, 'compile'):
                        if hasattr(pipeline, 'unet'):
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
                    logger.info("Model moved to CPU")
            except Exception as e:
                logger.warning(f"Failed to move model to CPU: {e}")
            
            # Delete pipeline reference
            del self.current_pipeline
            self.current_pipeline = None
            self.current_model_info = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear GPU cache multiple times for thorough cleanup
            if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
                for _ in range(3):  # Multiple clears for stubborn memory
                    clear_gpu_cache()
                
                # Log memory status after cleanup
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                memory_free = memory_total - memory_allocated
                logger.info(f"Memory after unload: {memory_allocated:.2f}GB allocated, {memory_free:.2f}GB free")
    
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
            ModelType.ANIMATEDIFF: 10.0,
            ModelType.FLUX: 6.0,  # FLUX models are slower but high quality
            ModelType.CUSTOM: 5.0,  # CUSTOM models like Bytedance Kolors, PixArt, etc.
        
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
    
    def force_memory_cleanup(self):
        """Force aggressive memory cleanup optimized for cloud GPUs."""
        logger.info("🧹 Forcing aggressive memory cleanup...")
        
        # Unload current model
        self._unload_current_model()
        
        # Clear model cache
        self.model_cache.clear()
        
        # Set optimized PyTorch memory allocation configuration for cloud GPUs
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:1024,roundup_power2_divisions:16'
        
        # Default cleanup rounds (can be overridden for cloud GPUs)
        cleanup_rounds = 10
        
        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            # Get initial memory stats
            initial_allocated = torch.cuda.memory_allocated() / 1024**3
            initial_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Before cleanup - Allocated: {initial_allocated:.2f}GB, Reserved: {initial_reserved:.2f}GB")
            
            # Cloud GPU specific optimizations
            if memory_total >= 48.0:  # Cloud GPU detected
                logger.info(f"🌩️  Cloud GPU detected ({memory_total:.0f}GB) - using enhanced cleanup")
                cleanup_rounds = 15  # More aggressive for cloud GPUs
            else:
                cleanup_rounds = 10  # Standard cleanup
            
            # Multiple rounds of ultra-aggressive cleanup
            for i in range(cleanup_rounds):
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear GPU cache
                clear_gpu_cache()
                
                # Additional PyTorch cleanup
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
                # Cloud GPU specific cleanup
                if memory_total >= 48.0:
                    # Additional operations for cloud GPUs
                    try:
                        # Reset memory allocator state
                        if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                            torch.cuda.reset_accumulated_memory_stats()
                        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                            torch.cuda.reset_peak_memory_stats()
                        
                        # Force memory pool reset
                        if hasattr(torch.cuda, 'memory'):
                            try:
                                torch.cuda.memory.empty_cache()
                            except:
                                pass
                        
                        # Additional synchronization for cloud GPUs
                        torch.cuda.synchronize()
                        
                    except Exception as cloud_cleanup_error:
                        # Don't fail the cleanup if advanced features aren't available
                        pass
                
                # Progressive delay for thorough cleanup
                import time
                time.sleep(0.05 + (i * 0.01))  # Increasing delay
            
            # Final reset and synchronization
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
                torch.cuda.synchronize()
                
                # For cloud GPUs, try to reset memory fraction
                if memory_total >= 48.0:
                    try:
                        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% for cloud GPUs
                        logger.info("🌩️  Set cloud GPU memory fraction to 95%")
                    except:
                        pass  # Ignore if not available
                
            except Exception as final_cleanup_error:
                logger.warning(f"Final cleanup operations failed: {final_cleanup_error}")
            
            # Log final memory status
            final_allocated = torch.cuda.memory_allocated() / 1024**3
            final_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_free = memory_total - final_allocated
            
            logger.info(f"After aggressive cleanup:")
            logger.info(f"  - Allocated: {final_allocated:.2f}GB (was {initial_allocated:.2f}GB)")
            logger.info(f"  - Reserved: {final_reserved:.2f}GB (was {initial_reserved:.2f}GB)")
            logger.info(f"  - Free: {memory_free:.2f}GB of {memory_total:.2f}GB total")
            logger.info(f"  - Freed: {initial_allocated - final_allocated:.2f}GB")
            
            # Warn if cleanup wasn't very effective
            if final_allocated > initial_allocated * 0.8:  # Less than 20% reduction
                logger.warning(f"⚠️  Cleanup was not very effective. Consider restarting Python.")
                if memory_total >= 48.0:  # Cloud GPU
                    logger.info("🌩️  For cloud GPUs, this may indicate heavy memory fragmentation.")
                    logger.info("💡 Try: Restart Python session or use smaller batch sizes.")
            
        else:
            # CPU-only cleanup
            import gc
            for i in range(cleanup_rounds):
                gc.collect()
            logger.info("💻 CPU-only memory cleanup completed")

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
