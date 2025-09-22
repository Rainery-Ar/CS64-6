import time
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
)
from torchvision import transforms
import evaluate

from loralayer import Linear  # 你之前自定义的 LoRA Linear 类


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
                merge_weights=False,
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
    # 1) Load dataset (MNIST)
    dataset = load_dataset("mnist",cache_dir="D:/hf_cache/datasets")

    # 2) Image transforms for ViT
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT 需要 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    def transform_examples(batch):
    # MNIST 是灰度图 (L)，ViT 需要 3 通道，所以要 repeat(3,1,1)
      images = [transform(img.convert("L")).repeat(3, 1, 1) for img in batch["image"]]
      labels = batch["label"]
      return {"pixel_values": images, "labels": labels}



    dataset = dataset.map(
    transform_examples,
    batched=True,
    remove_columns=["image", "label"],
    load_from_cache_file=False
)


  

    # 4) Load pretrained ViT model (10 labels for MNIST)
    model = ViTForImageClassification.from_pretrained(
        "facebook/deit-tiny-patch16-224",
        num_labels=10,
        ignore_mismatched_sizes=True,  # 处理分类头大小不匹配问题
        cache_dir="D:/hf_cache/models"
    )

    # 5) Apply LoRA
    model = apply_lora(model, r=2, alpha=8, dropout=0.1)

    # 6) Freeze all non-LoRA params
    freeze_non_lora_params(model)

    # 打印可训练参数比例
    trainable, total = 0, 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"Trainable params: {trainable} / {total} ({100*trainable/total:.2f}%)")

    # 7) Metrics
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    # 8) Training args
    training_args = TrainingArguments(
        learning_rate=5e-5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        save_strategy="epoch",
        fp16=True,
        report_to=[],
        dataloader_num_workers=4
    )

    # 9) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=None,
        compute_metrics=compute_metrics,
    )

    # 10) Train
    start = time.time()
    trainer.train()
    total_time = time.time() - start
    print(f"\nTotal training time: {total_time:.2f} sec ({total_time/60:.2f} min)")

    # 11) Evaluate
    results = trainer.evaluate()
    print("Test results:", results)


if __name__ == "__main__":
    main()
#training time:754.30 sec
#accuracy:0.8745