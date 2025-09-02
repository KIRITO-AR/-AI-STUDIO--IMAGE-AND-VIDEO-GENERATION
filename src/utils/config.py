"""
Configuration management for AI Generation Studio.
Handles loading and validation of application settings.
"""

import os
import configparser
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for AI models."""
    default_image_model: str
    default_video_model: str
    model_cache_dir: str
    max_memory_usage: float

@dataclass
class GenerationConfig:
    """Configuration for generation parameters."""
    default_steps: int
    default_guidance_scale: float
    default_width: int
    default_height: int
    default_video_frames: int
    default_video_fps: int

@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    use_xformers: bool
    use_tensorrt: bool
    enable_cpu_offload: bool
    use_attention_slicing: bool
    enable_model_cpu_offload: bool

@dataclass
class UIConfig:
    """Configuration for user interface."""
    theme: str
    show_advanced_options: bool
    auto_save_outputs: bool
    output_directory: str

@dataclass
class CloudConfig:
    """Configuration for cloud providers."""
    enable_cloud_gpu: bool
    preferred_provider: str
    max_cloud_cost_per_hour: float

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_level: str
    log_file: str
    enable_performance_metrics: bool

@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    allow_remote_access: bool
    require_api_key: bool
    max_requests_per_minute: int

class ConfigManager:
    """Manages application configuration loading and validation."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._find_config_file()
        self.config = configparser.ConfigParser()
        self._load_config()
        
        # Initialize configuration objects
        self.models = self._load_model_config()
        self.generation = self._load_generation_config()
        self.performance = self._load_performance_config()
        self.ui = self._load_ui_config()
        self.cloud = self._load_cloud_config()
        self.logging = self._load_logging_config()
        self.security = self._load_security_config()
    
    def _find_config_file(self) -> str:
        """Find the configuration file in the project."""
        possible_paths = [
            "configs/app.config",
            "../configs/app.config",
            "../../configs/app.config"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Could not find configuration file")
    
    def _load_config(self):
        """Load the configuration file."""
        try:
            self.config.read(self.config_file)
            logger.info(f"Configuration loaded from {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_model_config(self) -> ModelConfig:
        """Load model configuration."""
        section = self.config['models']
        return ModelConfig(
            default_image_model=section.get('default_image_model', 'runwayml/stable-diffusion-v1-5'),
            default_video_model=section.get('default_video_model', 'guoyww/animatediff-motion-adapter-v1-5-2'),
            model_cache_dir=section.get('model_cache_dir', './models_cache'),
            max_memory_usage=section.getfloat('max_memory_usage', 0.8)
        )
    
    def _load_generation_config(self) -> GenerationConfig:
        """Load generation configuration."""
        section = self.config['generation']
        return GenerationConfig(
            default_steps=section.getint('default_steps', 20),
            default_guidance_scale=section.getfloat('default_guidance_scale', 7.5),
            default_width=section.getint('default_width', 512),
            default_height=section.getint('default_height', 512),
            default_video_frames=section.getint('default_video_frames', 16),
            default_video_fps=section.getint('default_video_fps', 8)
        )
    
    def _load_performance_config(self) -> PerformanceConfig:
        """Load performance configuration."""
        section = self.config['performance']
        return PerformanceConfig(
            use_xformers=section.getboolean('use_xformers', True),
            use_tensorrt=section.getboolean('use_tensorrt', False),
            enable_cpu_offload=section.getboolean('enable_cpu_offload', True),
            use_attention_slicing=section.getboolean('use_attention_slicing', True),
            enable_model_cpu_offload=section.getboolean('enable_model_cpu_offload', False)
        )
    
    def _load_ui_config(self) -> UIConfig:
        """Load UI configuration."""
        section = self.config['ui']
        return UIConfig(
            theme=section.get('theme', 'dark'),
            show_advanced_options=section.getboolean('show_advanced_options', False),
            auto_save_outputs=section.getboolean('auto_save_outputs', True),
            output_directory=section.get('output_directory', './outputs')
        )
    
    def _load_cloud_config(self) -> CloudConfig:
        """Load cloud configuration."""
        section = self.config['cloud']
        return CloudConfig(
            enable_cloud_gpu=section.getboolean('enable_cloud_gpu', False),
            preferred_provider=section.get('preferred_provider', 'aws'),
            max_cloud_cost_per_hour=section.getfloat('max_cloud_cost_per_hour', 5.0)
        )
    
    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration."""
        section = self.config['logging']
        return LoggingConfig(
            log_level=section.get('log_level', 'INFO'),
            log_file=section.get('log_file', './logs/app.log'),
            enable_performance_metrics=section.getboolean('enable_performance_metrics', True)
        )
    
    def _load_security_config(self) -> SecurityConfig:
        """Load security configuration."""
        section = self.config['security']
        return SecurityConfig(
            allow_remote_access=section.getboolean('allow_remote_access', False),
            require_api_key=section.getboolean('require_api_key', False),
            max_requests_per_minute=section.getint('max_requests_per_minute', 60)
        )
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as a dictionary."""
        return {
            'models': self.models,
            'generation': self.generation,
            'performance': self.performance,
            'ui': self.ui,
            'cloud': self.cloud,
            'logging': self.logging,
            'security': self.security
        }
    
    def update_config(self, section: str, key: str, value: Any):
        """Update a configuration value."""
        if section not in self.config:
            self.config.add_section(section)
        
        self.config.set(section, key, str(value))
        
        # Reload the specific configuration object
        if section == 'models':
            self.models = self._load_model_config()
        elif section == 'generation':
            self.generation = self._load_generation_config()
        elif section == 'performance':
            self.performance = self._load_performance_config()
        elif section == 'ui':
            self.ui = self._load_ui_config()
        elif section == 'cloud':
            self.cloud = self._load_cloud_config()
        elif section == 'logging':
            self.logging = self._load_logging_config()
        elif section == 'security':
            self.security = self._load_security_config()
    
    def save_config(self):
        """Save the current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                self.config.write(f)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

# Global configuration instance
config_manager = None

def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager()
    return config_manager

def reload_config():
    """Reload the configuration from file."""
    global config_manager
    config_manager = ConfigManager()
    return config_manager