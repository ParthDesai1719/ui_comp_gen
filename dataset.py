from datasets import load_dataset
from data.dataset_formatter import format_fluent_dev_chat

def get_dataset():
    dataset = load_dataset("justmalhar/fluent-dev")
    formatted = format_fluent_dev_chat(dataset)
    print("Dataset loaded and formatted")
    return formatted
