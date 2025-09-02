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
from peft import LoraConfig, get_peft_model, TaskType

def main():
    # 1) Data: GLUE / MRPC
    dataset = load_dataset("glue", "mrpc")

    # 2) Tokenizer & base model
    model_name = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize_fn(batch):
        return tokenizer(batch["sentence1"], batch["sentence2"], truncation=True)

    encoded = dataset.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 3) Wrap with LoRA (r=2)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=2,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()  # shows % params being trained

    # 4) Metrics → Accuracy
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    # 5) TrainingArguments
    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir="./results_mrpc_lora",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,         # keep fast
        weight_decay=0.01,
        logging_dir="./logs_mrpc_lora",
        logging_steps=50,
        fp16=use_fp16,              # only if CUDA available
        report_to=[],               # disable wandb/tensorboard
    )

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7) Train with timing
    start = time.time()
    trainer.train()
    total = time.time() - start
    print(f"\n✅ Total training time: {total:.2f} seconds ({total/60:.2f} minutes)")

    # 8) Final evaluation
    results = trainer.evaluate()
    print("Validation Accuracy:", results["eval_accuracy"])

if __name__ == "__main__":
    main()
#training time:15.24 seconds
#accuracy:0.6838235294117647