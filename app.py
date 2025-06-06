# app.py

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
