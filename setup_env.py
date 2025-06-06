import torch

def check_gpu_setup():
    """Verify GPU availability and print setup information"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU detected! Training will be very slow.")
