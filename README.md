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

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

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
- **AnimateDiff**: Video generation from text prompts
- **ControlNet**: Enhanced control over generation process

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

## 🤝 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the documentation in `docs/`
- Review examples in `examples/`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Stability AI for Stable Diffusion
- Hugging Face for the Diffusers library
- The open-source AI community for continuous innovation
