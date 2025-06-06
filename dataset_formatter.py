from datasets import load_dataset, DatasetDict
from typing import Union

def format_fluent_dev_chat(dataset: Union[DatasetDict, dict]) -> dict:
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
        "validation": dataset["validation"].map(format_example, remove_columns=dataset["validation"].column_names)
    }

def load_and_format_dataset():
    dataset = load_dataset("justmalhar/fluent-dev")
    return format_fluent_dev_chat(dataset)
