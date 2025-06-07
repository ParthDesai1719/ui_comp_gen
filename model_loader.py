from unsloth import FastLanguageModel
import torch

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3.2-1b-instruct-bnb-4bit",
        max_seq_length=4096,
        dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    print("✓ LoRA adapters attached & model ready for fine-tuning!")
    return model, tokenizer
