import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
import time

# -----------------------------
# 1. Load dataset
# -----------------------------
dataset = load_dataset("ag_news")
train_list = [(item["label"], item["text"]) for item in dataset["train"]]
test_list = [(item["label"], item["text"]) for item in dataset["test"]]

# -----------------------------
# 2. Check labels
# -----------------------------
train_labels = [label for label, _ in train_list]
min_label, max_label = min(train_labels), max(train_labels)
num_classes = max_label + 1 if min_label == 0 else max_label
print(f"Number of classes: {num_classes}")

# -----------------------------
# 3. Load pretrained tokenizer & model
# -----------------------------
model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_classes
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# 4. Apply LoRA via PEFT
# -----------------------------
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=2,              # low-rank dimension
    lora_alpha=128,    # scaling factor
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)

# -----------------------------
# 5. Prepare DataLoader
# -----------------------------
def encode_batch(batch):
    texts = [text for _, text in batch]
    labels = [label - 1 if min_label == 1 else label for label, _ in batch]
    encodings = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", max_length=128
    )
    return encodings, torch.tensor(labels)

batch_size = 32
train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True, collate_fn=encode_batch)
test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False, collate_fn=encode_batch)

# -----------------------------
# 6. Training setup
# -----------------------------
criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
num_epochs =10

# -----------------------------
# 7. Training loop with timer
# -----------------------------
start_time = time.time()  # start timer

for epoch in range(num_epochs):
    model.train()
    total_acc, total_count = 0, 0
    for encodings, labels in train_loader:
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_acc += (logits.argmax(1) == labels).sum().item()
        total_count += labels.size(0)

    print(f"Epoch {epoch+1}, Train Accuracy: {total_acc/total_count:.4f}")

end_time = time.time()  # end timer
elapsed_time = end_time - start_time
print(f"\nTotal training time: {elapsed_time:.2f} seconds")

# -----------------------------
# 8. Evaluation
# -----------------------------
model.eval()
total_acc, total_count = 0, 0
with torch.no_grad():
    for encodings, labels in test_loader:
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        labels = labels.to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        total_acc += (logits.argmax(1) == labels).sum().item()
        total_count += labels.size(0)

print(f"Test Accuracy: {total_acc/total_count:.4f}")
#Total training time: 276.65 seconds
#Test Accuracy: 0.8874
