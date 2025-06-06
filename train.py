# train.py

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from config import max_seq_length, output_dir
from model_loader import load_model
from dataset_formatter import load_and_format_dataset

def train_model():
    model, tokenizer = load_model()
    dataset = load_and_format_dataset()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=1,
            learning_rate=2e-5,
            logging_steps=10,
            save_strategy="no",
            evaluation_strategy="epoch",
            bf16=False,
            fp16=True,
            lr_scheduler_type="cosine",
        ),
    )

    trainer.train()
