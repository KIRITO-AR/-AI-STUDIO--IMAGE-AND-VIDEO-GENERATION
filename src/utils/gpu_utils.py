"""
GPU detection and optimization utilities for AI Generation Studio.
Handles GPU detection, memory management, and performance optimization.
"""

import os
import logging
import platform
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import psutil

logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    """Information about a GPU device."""
    index: int
    name: str
    memory_total: int  # in MB
    memory_free: int   # in MB
    memory_used: int   # in MB
    utilization: float  # percentage
    temperature: Optional[float] = None
    driver_version: Optional[str] = None

@dataclass
class SystemInfo:
    """System information for optimization."""
    cpu_count: int
    ram_total: int  # in GB
    ram_available: int  # in GB
    platform: str
    python_version: str

class GPUDetector:
    """Detects and manages GPU resources."""
    
    def __init__(self):
        self.cuda_available = self._check_cuda()
        self.rocm_available = self._check_rocm()
        self.gpus = self._detect_gpus()
        self.system_info = self._get_system_info()
        
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            logger.warning("PyTorch not installed, CUDA detection skipped")
            return False
    
    def _check_rocm(self) -> bool:
        """Check if ROCm is available (AMD GPUs)."""
        try:
            import torch
            return hasattr(torch.version, 'hip') and torch.version.hip is not None
        except (ImportError, AttributeError):
            return False
    
    def _detect_gpus(self) -> List[GPUInfo]:
        """Detect available GPUs."""
        gpus = []
        
        if self.cuda_available:
            gpus.extend(self._detect_nvidia_gpus())
        
        if self.rocm_available:
            gpus.extend(self._detect_amd_gpus())
        
        if not gpus:
            logger.info("No compatible GPUs detected, will use CPU")
        
        return gpus
    
    def _detect_nvidia_gpus(self) -> List[GPUInfo]:
        """Detect NVIDIA GPUs using pynvml."""
        gpus = []
        
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get basic info
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Get memory info
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_total = memory_info.total // (1024 * 1024)  # Convert to MB
                memory_free = memory_info.free // (1024 * 1024)
                memory_used = memory_info.used // (1024 * 1024)
                
                # Get utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization = util.gpu
                except:
                    utilization = 0.0
                
                # Get temperature
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = None
                
                # Get driver version
                try:
                    driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                except:
                    driver_version = None
                
                gpu_info = GPUInfo(
                    index=i,
                    name=name,
                    memory_total=memory_total,
                    memory_free=memory_free,
                    memory_used=memory_used,
                    utilization=utilization,
                    temperature=temperature,
                    driver_version=driver_version
                )
                
                gpus.append(gpu_info)
                logger.info(f"Detected NVIDIA GPU {i}: {name} ({memory_total}MB)")
            
        except ImportError:
            logger.warning("pynvml not available, using torch for basic CUDA detection")
            # Fallback to basic torch detection
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        name = torch.cuda.get_device_name(i)
                        props = torch.cuda.get_device_properties(i)
                        memory_total = props.total_memory // (1024 * 1024)
                        
                        gpu_info = GPUInfo(
                            index=i,
                            name=name,
                            memory_total=memory_total,
                            memory_free=memory_total,  # Approximation
                            memory_used=0,
                            utilization=0.0
                        )
                        gpus.append(gpu_info)
                        logger.info(f"Detected CUDA GPU {i}: {name} ({memory_total}MB)")
            except Exception as e:
                logger.error(f"Failed to detect CUDA GPUs: {e}")
        
        except Exception as e:
            logger.error(f"Failed to detect NVIDIA GPUs: {e}")
        
        return gpus
    
    def _detect_amd_gpus(self) -> List[GPUInfo]:
        """Detect AMD GPUs using ROCm."""
        gpus = []
        
        try:
            import torch
            if hasattr(torch.version, 'hip') and torch.version.hip:
                for i in range(torch.cuda.device_count()):  # ROCm uses same API
                    name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    memory_total = props.total_memory // (1024 * 1024)
                    
                    gpu_info = GPUInfo(
                        index=i,
                        name=name,
                        memory_total=memory_total,
                        memory_free=memory_total,
                        memory_used=0,
                        utilization=0.0
                    )
                    gpus.append(gpu_info)
                    logger.info(f"Detected AMD GPU {i}: {name} ({memory_total}MB)")
        
        except Exception as e:
            logger.error(f"Failed to detect AMD GPUs: {e}")
        
        return gpus
    
    def _get_system_info(self) -> SystemInfo:
        """Get system information."""
        import sys
        
        ram_total = psutil.virtual_memory().total // (1024**3)  # GB
        ram_available = psutil.virtual_memory().available // (1024**3)  # GB
        
        return SystemInfo(
            cpu_count=psutil.cpu_count() or 4,
            ram_total=ram_total,
            ram_available=ram_available,
            platform=platform.system(),
            python_version=sys.version
        )
    
    def get_best_gpu(self) -> Optional[GPUInfo]:
        """Get the best available GPU for AI workloads."""
        if not self.gpus:
            return None
        
        # Sort by memory and utilization
        best_gpu = min(self.gpus, key=lambda gpu: (gpu.utilization, -gpu.memory_free))
        
        # Check if GPU has enough memory (minimum 4GB recommended)
        if best_gpu.memory_total < 4000:
            logger.warning(f"Best GPU {best_gpu.name} has only {best_gpu.memory_total}MB memory")
        
        return best_gpu
    
    def get_memory_recommendations(self) -> Dict[str, int]:
        """Get memory usage recommendations based on available hardware."""
        best_gpu = self.get_best_gpu()
        
        if not best_gpu:
            return {
                'batch_size': 1,
                'max_resolution': 512,
                'enable_cpu_offload': True,
                'use_attention_slicing': True
            }
        
        memory_gb = best_gpu.memory_total / 1024
        
        if memory_gb >= 24:  # High-end cards (RTX 4090, A100, etc.)
            return {
                'batch_size': 4,
                'max_resolution': 1024,
                'enable_cpu_offload': False,
                'use_attention_slicing': False
            }
        elif memory_gb >= 12:  # Mid-high range (RTX 3080Ti, RTX 4070Ti, etc.)
            return {
                'batch_size': 2,
                'max_resolution': 768,
                'enable_cpu_offload': False,
                'use_attention_slicing': True
            }
        elif memory_gb >= 8:   # Mid range (RTX 3070, RTX 4060Ti, etc.)
            return {
                'batch_size': 1,
                'max_resolution': 512,
                'enable_cpu_offload': True,
                'use_attention_slicing': True
            }
        else:  # Lower end
            return {
                'batch_size': 1,
                'max_resolution': 512,
                'enable_cpu_offload': True,
                'use_attention_slicing': True
            }
    
    def get_optimization_settings(self) -> Dict[str, bool]:
        """Get optimization settings based on hardware."""
        settings = {
            'use_torch_compile': False,
            'use_xformers': False,
            'use_tensorrt': False,
            'enable_sequential_cpu_offload': False,
            'enable_model_cpu_offload': False
        }
        
        # Check for xformers availability
        try:
            import xformers
            settings['use_xformers'] = True
            logger.info("xformers available, enabling attention optimization")
        except ImportError:
            logger.info("xformers not available")
        
        # Check for TensorRT
        try:
            import tensorrt
            if self.cuda_available:
                settings['use_tensorrt'] = True
                logger.info("TensorRT available, can enable for inference acceleration")
        except ImportError:
            logger.info("TensorRT not available")
        
        # Enable CPU offloading based on GPU memory
        best_gpu = self.get_best_gpu()
        if best_gpu and best_gpu.memory_total < 8000:  # Less than 8GB
            settings['enable_sequential_cpu_offload'] = True
            logger.info("Enabling CPU offloading due to limited GPU memory")
        
        return settings

class PerformanceMonitor:
    """Monitors GPU and system performance during generation."""
    
    def __init__(self, gpu_detector: GPUDetector):
        self.gpu_detector = gpu_detector
        self.monitoring = False
        self.stats = []
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.stats = []
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self) -> Dict:
        """Stop monitoring and return statistics."""
        self.monitoring = False
        
        if not self.stats:
            return {}
        
        # Calculate averages
        avg_stats = {}
        if self.stats:
            for key in self.stats[0].keys():
                avg_stats[f'avg_{key}'] = sum(stat[key] for stat in self.stats) / len(self.stats)
        
        logger.info("Performance monitoring stopped")
        return avg_stats
    
    def collect_stats(self) -> Dict:
        """Collect current performance statistics."""
        stats = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3)
        }
        
        # Add GPU stats
        for i, gpu in enumerate(self.gpu_detector.gpus):
            try:
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.index)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                stats[f'gpu_{i}_memory_used_percent'] = (memory_info.used / memory_info.total) * 100
                stats[f'gpu_{i}_utilization'] = util.gpu
                
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    stats[f'gpu_{i}_temperature'] = temp
                except:
                    pass
                    
            except:
                # Fallback to basic stats
                stats[f'gpu_{i}_available'] = True
        
        if self.monitoring:
            self.stats.append(stats)
        
        return stats

def get_device_info() -> Tuple[str, GPUDetector]:
    """Get the best device to use and detector instance."""
    detector = GPUDetector()
    
    if detector.cuda_available and detector.gpus:
        best_gpu = detector.get_best_gpu()
        if best_gpu:
            device = f"cuda:{best_gpu.index}"
            logger.info(f"Using GPU: {best_gpu.name} (CUDA:{best_gpu.index})")
        else:
            device = "cpu"
            logger.info("Using CPU (no suitable GPU found)")
    elif detector.rocm_available and detector.gpus:
        best_gpu = detector.get_best_gpu()
        if best_gpu:
            device = f"cuda:{best_gpu.index}"  # ROCm uses same API
            logger.info(f"Using GPU: {best_gpu.name} (ROCm:{best_gpu.index})")
        else:
            device = "cpu"
            logger.info("Using CPU (no suitable GPU found)")
    else:
        device = "cpu"
        logger.info("Using CPU (no compatible GPU found)")
    
    return device, detector

def optimize_torch_settings(detector: GPUDetector):
    """Optimize PyTorch settings based on hardware."""
    try:
        import torch
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory allocation strategy
        if detector.cuda_available:
            # Use memory fraction if GPU memory is limited
            best_gpu = detector.get_best_gpu()
            if best_gpu and best_gpu.memory_total < 8000:
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        logger.info("PyTorch optimization settings applied")
        
    except Exception as e:
        logger.warning(f"Failed to apply PyTorch optimizations: {e}")

def clear_gpu_cache():
    """Clear GPU memory cache."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    except Exception as e:
        logger.warning(f"Failed to clear GPU cache: {e}")