import time
import numpy as np
import torch
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
    print("\n===== Running QNLI | Base =====")

    # 1) Dataset
    dataset = load_dataset("glue", "qnli")

    # 2) Tokenizer & model
    model_name = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize_fn(batch):
        return tokenizer(batch["question"], batch["sentence"], truncation=True)

    encoded = dataset.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 3) Metric
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    # 4) TrainingArguments
    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs_qnli_base",
        logging_steps=100,
        fp16=use_fp16,
        report_to=[],
        save_strategy="no"
    )

    # 5) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6) Train
    start = time.time()
    trainer.train()
    print(f"✅ Training time: {time.time()-start:.2f} sec")

    # 7) Evaluate
    results = trainer.evaluate()
    print("🔎 Final QNLI Results:", results)

if __name__ == "__main__":
    main()
#training time:1046.93 sec
#accuracy:0.9412410763316859