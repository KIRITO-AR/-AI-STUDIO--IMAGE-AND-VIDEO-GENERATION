import sys

# Redirect output to a file
with open('installation_check_result.txt', 'w') as f:
    f.write("Python version: " + sys.version + "\n")
    
    try:
        import torch
        f.write("PyTorch imported successfully\n")
        f.write("PyTorch version: " + str(torch.__version__) + "\n")
        f.write("CUDA available: " + str(torch.cuda.is_available()) + "\n")
        if torch.cuda.is_available():
            f.write("CUDA version: " + str(torch.version.cuda) + "\n")
            f.write("Number of GPUs: " + str(torch.cuda.device_count()) + "\n")
            for i in range(torch.cuda.device_count()):
                f.write(f"GPU {i}: {torch.cuda.get_device_name(i)}\n")
    except ImportError as e:
        f.write("Failed to import PyTorch: " + str(e) + "\n")
    
    try:
        import diffusers
        f.write("Diffusers imported successfully\n")
        f.write("Diffusers version: " + str(diffusers.__version__) + "\n")
    except ImportError as e:
        f.write("Failed to import diffusers: " + str(e) + "\n")
    
    f.write("Installation check completed.\n")