import sys
print("Python version:", sys.version)

try:
    import torch
    print("PyTorch imported successfully")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Number of GPUs:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError as e:
    print("Failed to import PyTorch:", e)

try:
    import diffusers
    print("Diffusers imported successfully")
    print("Diffusers version:", diffusers.__version__)
except ImportError as e:
    print("Failed to import diffusers:", e)