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

from loralayer import Linear  # Your custom LoRA Linear class


# -----------------------------
# Apply LoRA recursively
# -----------------------------
def apply_lora(model, r=2, alpha=8, dropout=0.1):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            new_module = Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                fan_in_fan_out=False,
                merge_weights=False
            )
            new_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_module.bias.data = module.bias.data.clone()
            setattr(model, name, new_module)
        else:
            apply_lora(module, r=r, alpha=alpha, dropout=dropout)
    return model


# -----------------------------
# Freeze all non-LoRA parameters
# -----------------------------
def freeze_non_lora_params(model):
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True


# -----------------------------
# Main function
# -----------------------------
def main():
    # 1) Load dataset
    dataset = load_dataset("glue", "cola")

    # 2) Tokenizer
    model_name = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize_fn(batch):
        return tokenizer(batch["sentence"], truncation=True)

    encoded = dataset.map(tokenize_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3) Load pretrained model (binary classification)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 4) Apply LoRA (r=2)
    model = apply_lora(model, r=2, alpha=8, dropout=0.1)

    # 5) Freeze all non-LoRA parameters
    freeze_non_lora_params(model)

    # Print trainable parameters
    trainable, total = 0, 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"Trainable params: {trainable} / {total} ({100*trainable/total:.2f}%)")

    # 6) Metrics
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    # 7) Training arguments
    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        fp16=use_fp16,
        report_to=[],
    )

    # 8) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 9) Train
    start = time.time()
    trainer.train()
    total_time = time.time() - start
    print(f"\nTotal training time: {total_time:.2f} sec ({total_time/60:.2f} min)")

    # 10) Evaluate
    results = trainer.evaluate()
    print("Validation results:", results)


if __name__ == "__main__":
    main()
#training time:51.41 sec (0.86 min)
#accuracy:0.6912751677852349