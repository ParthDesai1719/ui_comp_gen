%%capture
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install trl peft accelerate bitsandbytes datasets pandas scikit-learn gradio

from torch import __version__ as torch_version
from packaging.version import Version as V
xformers_version = "xformers==0.0.27" if V(torch_version) < V("2.4.0") else "xformers"
!pip install --no-deps {xformers_version}

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import gradio as gr
from IPython.display import HTML

#checking gpu setup
def check_gpu_setup():
    """Verify GPU availability and print setup information"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: No GPU detected! Training will be very slow.")

check_gpu_setup()
