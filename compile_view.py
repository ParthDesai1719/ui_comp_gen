# === ui_comp_gen/config.py ===
model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
dataset_name = "justmalhar/fluent-dev"
max_seq_length = 4096
output_dir = "fluent-ui-generator"


# === ui_comp_gen/setup_env.py ===
import torch
from packaging.version import Version as V
import os

def install_and_check():
    os.system('pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
    os.system('pip install trl peft accelerate bitsandbytes datasets pandas scikit-learn gradio')
    torch_version = torch.__version__
    xformers = "xformers==0.0.27" if V(torch_version) < V("2.4.0") else "xformers"
    os.system(f"pip install --no-deps {xformers}")
    print("✓ Dependencies installed.")


# === ui_comp_gen/model_loader.py ===
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


# === ui_comp_gen/dataset_formatter.py ===
def format_fluent_dev_chat(dataset):
    def format_example(example):
        return {
            "text": f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Generate a {example.get('category', 'component')} component with:
- Tags: {example.get('tags', [])}
- Colors: {example.get('colors', [])}
- Description: {example.get('description', '').strip()}
{example.get('instruction', '').strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{example.get('code', '').strip()}<|eot_id|>"""
        }
    return {
        "train": dataset["train"].map(format_example, remove_columns=dataset["train"].column_names),
        "validation": dataset["validation"].map(format_example, remove_columns=dataset["validation"].column_names),
    }


# === ui_comp_gen/dataset.py ===
from datasets import load_dataset
from dataset_formatter import format_fluent_dev_chat

def get_dataset():
    dataset = load_dataset("justmalhar/fluent-dev")
    formatted = format_fluent_dev_chat(dataset)
    print("✓ Dataset loaded and formatted")
    return formatted


# === ui_comp_gen/train.py ===
from transformers import TrainingArguments
from trl import SFTTrainer
from config import output_dir, max_seq_length
from dataset import get_dataset
from model_loader import load_model
from transformers import DataCollatorForSeq2Seq

def train_model():
    model, tokenizer = load_model()
    dataset = get_dataset()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer),
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            num_train_epochs=3,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            fp16=True,
            report_to="none",
        ),
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✓ Training complete and model saved.")


# === ui_comp_gen/utils.py ===
def clean_response(decoded):
    for tok in ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|eos|>"]:
        decoded = decoded.replace(tok, "")
    decoded = decoded.strip()
    lines = decoded.splitlines()
    cleaned = [line for line in lines if not line.strip().lower().startswith("generate a")]
    return "\n".join(cleaned)


# === ui_comp_gen/app.py ===
import gradio as gr
from model_loader import load_model
from utils import clean_response

model, tokenizer = load_model()

def generate_component(prompt):
    full_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=512,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if "<|start_header_id|>assistant<|end_header_id|>" in decoded:
        decoded = decoded.split("<|start_header_id|>assistant<|end_header_id|>")[1]
    if "<|eot_id|>" in decoded:
        decoded = decoded.split("<|eot_id|>")[0]
    return clean_response(decoded)

gr.Interface(
    fn=generate_component,
    inputs=gr.Textbox(lines=6, label="Describe your component"),
    outputs=gr.Code(label="Generated JSX"),
    title="🧠 UI Component Generator (LLaMA 3)"
).launch(share=True)
