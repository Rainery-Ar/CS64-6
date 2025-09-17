import torch
import time
from torch import nn
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# -----------------------------
# 1. Load dataset
# -----------------------------
train_dataset, test_dataset = AG_NEWS()
train_list = list(train_dataset)
test_list = list(test_dataset)

# -----------------------------
# 2. Check label range
# -----------------------------
train_labels = [label for label, _ in train_list]
min_label, max_label = min(train_labels), max(train_labels)
print(f"Train labels min={min_label}, max={max_label}")

num_classes = max_label + 1 if min_label == 0 else max_label  # adjust to 0-based
pad_index = 0

# -----------------------------
# 3. Compute vocab_size safely
# -----------------------------
all_tokens = torch.cat([text for _, text in train_list])
vocab_size = int(all_tokens.max()) + 1
print(f"Vocab size: {vocab_size}")

# -----------------------------
# 4. Collate function
# -----------------------------
def collate_batch(batch):
    texts, labels = [], []
    for label, text in batch:
        texts.append(text.clone())

        # Safe label mapping to 0-based
        if min_label == 1:
            mapped_label = label - 1
        else:
            mapped_label = label
        labels.append(mapped_label)

    texts = pad_sequence(texts, batch_first=True, padding_value=pad_index)
    labels = torch.tensor(labels, dtype=torch.long)
    return texts, labels

batch_size = 32
train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# -----------------------------
# 5. Small Bidirectional LSTM model
# -----------------------------
class SmallBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=50, hidden_dim=128, output_dim=num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 because bidirectional

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        # Concatenate last hidden states from both directions
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden_cat = self.dropout(hidden_cat)
        return self.fc(hidden_cat)

device = torch.device("cpu")
model = SmallBiLSTM(vocab_size=vocab_size).to(device)

# -----------------------------
# 6. Training
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs =3

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    total_acc, total_count = 0, 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(texts)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_acc += (output.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
    print(f"Epoch {epoch+1}, Train Accuracy: {total_acc/total_count:.4f}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal training time: {elapsed_time:.2f} seconds")

# -----------------------------
# 7. Evaluation
# -----------------------------
model.eval()
total_acc, total_count = 0, 0
with torch.no_grad():
    for texts, labels in test_loader:
        texts, labels = texts.to(device), labels.to(device)
        output = model(texts)
        total_acc += (output.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
print(f"Test Accuracy: {total_acc/total_count:.4f}")
#Total training time: 249.25 seconds
#Test Accuracy: 0.7238
