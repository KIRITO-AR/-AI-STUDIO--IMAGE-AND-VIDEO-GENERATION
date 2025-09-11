# AI Image and Video Generation Studio

A comprehensive AI-powered application for generating high-quality images and videos from text prompts using state-of-the-art models like Stable Diffusion and AnimateDiff.

## 🎯 Features

### Core Capabilities
- **Text-to-Image Generation**: Create stunning images from descriptive text prompts
- **Text-to-Video Generation**: Generate short video clips using AnimateDiff
- **Batch Processing**: Process multiple prompts simultaneously
- **GPU Optimization**: Intelligent GPU detection and performance optimization
- **Cloud Integration**: Support for AWS, Google Cloud, and Azure GPU instances

### User Interfaces
- **Streamlit Prototype**: Quick and easy web-based interface for rapid prototyping
- **Professional Desktop App**: Full-featured PyQt6/PySide6 application with advanced controls

### Advanced Features
- **Model Management**: Easy switching between different AI models
- **Output Customization**: Control resolution, steps, guidance scale, and more
- **Real-time Preview**: See generation progress in real-time
- **Export Options**: Multiple format support (PNG, JPG, MP4, GIF)

## 🏗️ Project Structure

```
IMAGE-AND-VIDEO-GENERATION/
├── src/
│   ├── core/           # Core AI model integration
│   ├── models/         # Model management and loading
│   ├── ui/            # User interface components
│   ├── utils/         # Utility functions and helpers
│   └── cloud/         # Cloud provider integrations
├── tests/             # Unit and integration tests
├── configs/           # Configuration files
├── docs/             # Documentation
├── assets/           # Static assets and resources
├── examples/         # Example scripts and notebooks
└── requirements.txt  # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended)
- At least 8GB VRAM for optimal performance

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd IMAGE-AND-VIDEO-GENERATION
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: If you encounter import errors like "None of PyTorch, TensorFlow >= 2.0, or Flax have been found", make sure you're using the correct Python environment where the packages are installed. The virtual environment approach above ensures all dependencies are correctly installed and accessible.

If you're using an MSYS2/MinGW environment, you may need to use:
```bash
python -m pip install -r requirements.txt
```

For more detailed troubleshooting, check the [SOLUTION_SUMMARY.md](SOLUTION_SUMMARY.md) and [FIX_INSTRUCTIONS.md](FIX_INSTRUCTIONS.md) files.

### Running the Application

**Streamlit Prototype (Quick Start):**
```bash
streamlit run src/ui/streamlit_app.py
```

**Desktop Application:**
```bash
python src/ui/desktop_app.py
```

## 🧠 AI Models

### Supported Models
- **Stable Diffusion 1.5**: Classic and reliable image generation
- **Stable Diffusion XL**: Higher resolution and improved quality
- **AnimateDiff**: Video generation from text prompts (Basic video generation)
- **ModelScope T2V**: Advanced text-to-video model with better quality and consistency
- **Zeroscope V2**: High-quality, watermark-free video generation models
- **ControlNet**: Enhanced control over generation process

### Video Generation Models
The application now supports multiple video generation models with different capabilities:

1. **AnimateDiff**: The default video generation model, good for basic animations
2. **ModelScope T2V**: Advanced model with better temporal consistency
3. **Zeroscope V2 576w**: High-quality video generation with specific resolution requirements
4. **Zeroscope V2 XL**: Video upscaling model for higher resolution output

### Model Management
Models are automatically downloaded and cached on first use. The application intelligently manages memory and switches between models as needed.

## ⚙️ Configuration

### GPU Configuration
The application automatically detects available GPUs and optimizes settings accordingly:
- NVIDIA GPUs with CUDA
- AMD GPUs with ROCm (experimental)
- CPU fallback for systems without dedicated GPUs

### Cloud Configuration
Support for cloud GPU instances:
- AWS EC2 with GPU instances
- Google Cloud Platform with TPUs/GPUs
- Azure with NVIDIA GPU VMs

## 🔧 Development

### Setting up Development Environment
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📦 Deployment

### Creating Executable
```bash
# Build standalone executable
pyinstaller configs/pyinstaller.spec
```

### Docker Deployment
```bash
# Build Docker image
docker build -t ai-generation-studio .

# Run container
docker run -p 8501:8501 ai-generation-studio
```

## 🎨 Examples

Check the `examples/` directory for:
- Basic image generation scripts
- Video creation examples
  - Simple video generation with AnimateDiff
  - Advanced video generation with ModelScope T2V
  - High-quality video generation with Zeroscope
  - Video upscaling techniques
  - Video editing and post-processing
- Batch processing workflows
- Advanced customization techniques

## 📋 System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- 4GB GPU VRAM (or CPU fallback)
- 10GB free disk space

### Recommended Requirements
- Python 3.10+
- 16GB+ RAM
- NVIDIA RTX 3070 or better (8GB+ VRAM)
- 50GB+ free disk space (for model storage)

### Video Generation Requirements
- **Basic Video**: 6GB+ VRAM (AnimateDiff)
- **High-Quality Video**: 8GB+ VRAM (Zeroscope)
- **Video Upscaling**: 12GB+ VRAM (Zeroscope XL)

## 🤝 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the documentation in `docs/`
- Review examples in `examples/`

## 🛠️ Troubleshooting

### Common Issues and Solutions

**Import Errors ("None of PyTorch, TensorFlow >= 2.0, or Flax have been found")**
- Make sure you're using the correct Python environment where packages are installed
- Activate the virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
- Check that packages are installed: `pip list | grep torch`

**Python Version Mismatch**
- Ensure `python` and `pip` are using the same Python version
- Use `python -m pip` instead of `pip` to ensure version consistency

**MSYS2/MinGW Environment Issues**
- Create a virtual environment to avoid system package restrictions
- Use `python -m pip install` instead of `pip install`

**ONNX Runtime Errors on Windows**
- The application already includes environment variables to disable ONNX
- If issues persist, install Microsoft Visual C++ Redistributable
- Alternatively, uninstall onnxruntime: `pip uninstall onnxruntime onnxruntime-gpu`

For more detailed troubleshooting, check the [SOLUTION_SUMMARY.md](SOLUTION_SUMMARY.md) and [FIX_INSTRUCTIONS.md](FIX_INSTRUCTIONS.md) files.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Stability AI for Stable Diffusion
- Hugging Face for the Diffusers library
- The open-source AI community for continuous innovation
