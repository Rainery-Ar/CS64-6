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
    print("\n===== Running RTE | Base =====")

    dataset = load_dataset("glue", "rte")

    model_name = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize_fn(batch):
        return tokenizer(batch["sentence1"], batch["sentence2"], truncation=True)

    encoded = dataset.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs_rte_base",
        logging_steps=100,
        fp16=use_fp16,
        report_to=[],
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    start = time.time()
    trainer.train()
    print(f"✅ Training time: {time.time()-start:.2f} sec")

    results = trainer.evaluate()
    print("🔎 Final RTE Results:", results)

if __name__ == "__main__":
    main()
#training time:32.35 sec
#accuracy:0.6750902527075813