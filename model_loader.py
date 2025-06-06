from unsloth import FastLanguageModel
from config import model_name, max_seq_length
import torch

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto",
    )
    print("✓ Model loaded successfully!")
    return model, tokenizer
model, tokenizer = load_model()
