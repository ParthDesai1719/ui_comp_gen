##🧠 UI Component Generator (LLaMA 3.2 - Fine-Tuned)
This project fine-tunes the [LLaMA 3.2 1B Instruct](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-bnb-4bit) model to generate React/JSX UI components from natural language descriptions. Built with [Unsloth](https://github.com/unslothai/unsloth), [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), and [Gradio](https://gradio.app).

##🗂️ Project Structure
ui_comp_gen/
├── config.py # Model and dataset configuration
├── setup_env.py # Install dependencies and check environment
├── model_loader.py # Load LLaMA 3.2 model using Unsloth
├── dataset_formatter.py # Prepare FluentDev dataset for training
├── dataset.py # Load and inspect dataset
├── utils.py # Helper functions (e.g., output cleaning)
├── train.py # Fine-tune the model
├── app.py # Gradio app for generation
├── requirements.txt # All dependencies
└── README.md # This file

->Installation
```bash
pip install -r requirements.txt

->Training
bash
python setup_env.py
python train.py

->Run the App
Edit
python app.py

->Dataset
We use justmalhar/fluent-dev, which contains component descriptions and their corresponding JSX code, tagged by UI properties.

->Acknowledgements
Unsloth
HuggingFace Transformers
Fluent-Dev Dataset

->Author
Parth Desai
🔗 GitHub: ParthDesai1719
