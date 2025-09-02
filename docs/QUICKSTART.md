# Quick Start Guide

Welcome to AI Generation Studio! This guide will help you get up and running quickly.

## Prerequisites

- Python 3.8 or higher
- At least 8GB RAM (16GB recommended)
- NVIDIA GPU with 4GB+ VRAM (optional but recommended)
- 10GB free disk space

## Installation

### 1. Quick Setup
```bash
python setup.py
```

This will:
- Check your system requirements
- Install all dependencies
- Create necessary directories
- Test basic functionality

### 2. Manual Setup
If you prefer manual setup:

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir outputs models_cache logs
```

## First Run

### Launch Streamlit Interface
```bash
python launch.py --streamlit
```

Then open your browser to: http://localhost:8501

### Run Examples
```bash
# Simple generation
python examples/simple_generation.py

# Batch generation
python examples/batch_generation.py
```

## Basic Usage

### 1. Load a Model
- Use the sidebar to select and load a model
- Start with "Stable Diffusion 1.5" for best compatibility

### 2. Generate Images
- Enter your prompt in the text area
- Adjust settings if needed
- Click "Generate Images"

### 3. Save Results
- Click "Save Images" to save to the outputs folder
- Images include metadata with generation parameters

## Tips for Better Results

### Prompt Writing
- Be specific and descriptive
- Use artistic styles: "digital art", "oil painting", "photography"
- Add quality modifiers: "highly detailed", "8k", "masterpiece"

### Negative Prompts
- Add unwanted elements: "blurry, low quality, deformed"
- Use for style control: "cartoon" if you want realistic

### Settings
- **Steps**: 20-30 for good quality, 50+ for best quality
- **Guidance Scale**: 7-12 for most prompts
- **Resolution**: Start with 512x512, increase if GPU allows

## Troubleshooting

### Common Issues

**"No model loaded"**
- Load a model using the sidebar
- Check internet connection for first-time downloads

**"CUDA out of memory"**
- Reduce image size
- Lower the number of inference steps
- Enable CPU offloading in advanced settings

**Slow generation**
- GPU recommended for reasonable speed
- CPU generation is very slow but works

**Installation errors**
- Make sure Python 3.8+ is installed
- Try updating pip: `python -m pip install --upgrade pip`
- Install Visual Studio Build Tools if on Windows

### Getting Help

1. Check the logs in the `logs/` folder
2. Look at the console output for error messages
3. Try the examples to verify installation
4. Check system requirements

## Model Information

### Stable Diffusion 1.5
- **Best for**: General purpose, versatile
- **Memory**: ~4GB VRAM
- **Resolution**: Up to 768x768

### Stable Diffusion XL
- **Best for**: High quality, detailed images
- **Memory**: ~8GB VRAM
- **Resolution**: Up to 1024x1024

### AnimateDiff
- **Best for**: Short video clips
- **Memory**: ~6GB VRAM
- **Output**: 8-32 frame videos

## Advanced Features

### Batch Generation
- Generate multiple images from different prompts
- Perfect for testing variations
- Automatic saving and organization

### Video Generation
- Available with AnimateDiff model
- Generate 8-32 frame video clips
- Export as MP4 or GIF

### Custom Settings
- Adjust scheduler for different styles
- Control seed for reproducible results
- Fine-tune guidance and steps

## Next Steps

- Explore different models and settings
- Try the batch generation for multiple variations
- Experiment with different prompt styles
- Check out the desktop interface (coming soon)

Happy generating! 🎨