from datasets import load_dataset

def get_dataset(name="justmalhar/fluent-dev"):
    return load_dataset(name)
