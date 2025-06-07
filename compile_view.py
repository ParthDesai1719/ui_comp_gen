# === INSTALLS (uncomment if needed for Colab/local) ===
!pip install --upgrade pip
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -U trl peft accelerate bitsandbytes datasets pandas scikit-learn gradio transformers

# === CONFIG ===
model_name = "unsloth/Llama-3-2B-Instruct-bnb-4bit"
dataset_name = "justmalhar/fluent-dev"
max_seq_length = 3072  # T4 safe
output_dir = "fluent-ui-generator"

# === IMPORTS ===
import torch
import gradio as gr
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer

# === FORMAT DATASET ===
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

# === LOAD DATASET ===
def get_dataset():
    dataset = load_dataset(dataset_name)
    formatted = format_fluent_dev_chat(dataset)
    print("✓ Dataset loaded and formatted")
    return formatted

# === LOAD MODEL ===
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

    print("✓ LoRA adapters attached & model ready!")
    return model, tokenizer

# === TRAIN MODEL ===
def train_model(model, tokenizer, dataset):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=4096,
        data_collator=DataCollatorForSeq2Seq(tokenizer),
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            num_train_epochs=3,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            report_to="none",
        ),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✓ Training complete and model saved.")

# === CLEAN OUTPUT ===
def clean_response(decoded):
    for tok in ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|eos|>"]:
        decoded = decoded.replace(tok, "")
    decoded = decoded.strip()
    lines = decoded.splitlines()
    cleaned = [line for line in lines if not line.strip().lower().startswith("generate a")]
    return "\n".join(cleaned)

# === GENERATE COMPONENT ===
def generate_component(prompt):
    formatted_prompt = f"<|begin_of_text|><|user|>\n{prompt.strip()}\n<|assistant|>"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=768,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    start = decoded.find("<|assistant|>") + len("<|assistant|>")
    end = decoded.find("<|eot_id|>", start)
    result = decoded[start:end].strip() if end != -1 else decoded[start:].strip()
    return clean_response(result)

# === MAIN ===
if __name__ == "__main__":
    # Load model and dataset
    model, tokenizer = load_model()
    dataset = get_dataset()

    # Train the model (comment out after training to avoid re-training)
    # train_model(model, tokenizer, dataset)

    # Launch Gradio app
    gr.Interface(
        fn=generate_component,
        inputs=gr.Textbox(lines=6, label="Describe your component"),
        outputs=gr.Code(label="Generated JSX"),
        title="🧠 UI Component Generator (LLaMA 3)"
    ).launch(share=True)
