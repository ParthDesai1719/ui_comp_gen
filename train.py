from config.config import output_dir
from training.model_loader import load_model
from data.dataset import get_dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer

model, tokenizer = load_model()
dataset = get_dataset()

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
