# Fix for PyTorch and Diffusers Import Issues

## Problem Analysis

The error messages indicate that PyTorch and TensorFlow/Flax are not found, which prevents the AI models from loading. However, from our investigation, we can see that:

1. The required packages are installed in a virtual environment
2. The packages include:
   - PyTorch (torch-2.8.0)
   - Diffusers (diffusers-0.35.1)
   - Transformers (transformers-4.56.1)
   - And all other required dependencies

## Root Cause

The issue is caused by a Python environment mismatch:
- Your `python` command uses Python 3.12.11
- Your `pip` command installs packages for Python 3.13
- This mismatch prevents Python from finding the installed packages

## Solution

### Step 1: Use the Virtual Environment

We've already created a virtual environment with all required packages. To use it:

```bash
# Activate the virtual environment
.\venv\Scripts\activate

# Now run your Python scripts
python launch.py --streamlit
```

### Step 2: Verify Installation (if you can see output)

In the virtual environment, run:

```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

### Step 3: Alternative Solution - Install for System Python

If you prefer to use your system Python directly:

```bash
# Use the specific Python version to install packages
python -m pip install torch torchvision diffusers transformers accelerate safetensors
```

Note: You might need to add `--user` flag if you get permission errors:
```bash
python -m pip install --user torch torchvision diffusers transformers accelerate safetensors
```

## Running the Application

After fixing the environment issue, you can run:

1. **Streamlit UI**:
   ```bash
   python launch.py --streamlit
   ```

2. **Examples**:
   ```bash
   python examples/simple_generation.py
   ```

3. **Video Generation**:
   ```bash
   python examples/video_generation.py
   ```

## Windows-Specific Notes

If you encounter ONNX runtime errors on Windows:
1. The code already includes environment variables to disable ONNX
2. If issues persist, install Microsoft Visual C++ Redistributable

## Cloud GPU Setup

For cloud GPU usage:
1. Run `python start_cloud_gpu.py` for automatic setup
2. Check `VULTR_SETUP.md` for manual instructions

## Troubleshooting

1. **Import errors**: Ensure you're using the correct Python environment
2. **CUDA issues**: The application will automatically fall back to CPU
3. **Memory errors**: Use lower resolution settings or enable CPU offloading

The application is designed to work gracefully with or without GPU acceleration.