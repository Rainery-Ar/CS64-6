import time
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate

def main():
    dataset = load_dataset("glue", "qqp")
    model_name = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize_fn(batch):
        return tokenizer(batch["question1"], batch["question2"], truncation=True)

    encoded = dataset.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir="./results_qqp",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs_qqp",
        logging_steps=50,
        fp16=use_fp16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"].shuffle(seed=42).select(range(20000)),  # subset for speed
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    print(f"\n✅ Training time: {(time.time()-start)/60:.2f} min")

    results = trainer.evaluate()
    print("Validation Accuracy:", results["eval_accuracy"])

if __name__ == "__main__":
    main()
#training time:2.99 min
#accuracy:0.8722977986643582