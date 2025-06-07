import torch
from unsloth import FastLanguageModel
import gradio as gr
from inference.utils import clean_response

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.2-1b-instruct-bnb-4bit",
    max_seq_length=4096,
    dtype=torch.float16,
    load_in_4bit=True,
    device_map="auto",
)
print("✓ Model loaded successfully!")

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

gr.Interface(
    fn=generate_component,
    inputs=gr.Textbox(lines=6, label="Describe your component"),
    outputs=gr.Code(label="Generated JSX"),
    title="🧠 UI Component Generator (LLaMA 3)"
).launch(share=True)
