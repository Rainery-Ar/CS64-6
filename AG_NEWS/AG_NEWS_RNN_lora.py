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
test_list  = list(test_dataset)

# -----------------------------
# 2. Check label range
# -----------------------------
train_labels = [label for label, _ in train_list]
min_label, max_label = min(train_labels), max(train_labels)
print(f"Train labels min={min_label}, max={max_label}")

num_classes = max_label + 1 if min_label == 0 else max_label  # adjust to 0-based
pad_index = 0
MAX_LEN = 256  # truncate to speed up training

# -----------------------------
# 3. Compute vocab_size safely
# -----------------------------
all_tokens = torch.cat([text for _, text in train_list])
vocab_size = int(all_tokens.max()) + 1
print(f"Vocab size: {vocab_size}")

# -----------------------------
# 4. Collate function (truncate + pad)
# -----------------------------
def collate_batch(batch):
    texts, labels = [], []
    for label, text in batch:
        if text.numel() > MAX_LEN:
            text = text[:MAX_LEN]
        texts.append(text.clone())
        mapped_label = label - 1 if min_label == 1 else label
        labels.append(mapped_label)
    texts = pad_sequence(texts, batch_first=True, padding_value=pad_index)
    labels = torch.tensor(labels, dtype=torch.long)
    return texts, labels

# use batch_size=32 as requested
batch_size = 32
train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True,  collate_fn=collate_batch)
test_loader  = DataLoader(test_list,  batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# -----------------------------
# 5. LoRA building blocks
# -----------------------------
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=2, alpha=1.0, init_frozen="zeros"):
        super().__init__()
        self.r = r
        self.scaling = alpha / r

        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        if init_frozen == "zeros":
            nn.init.zeros_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        self.A = nn.Parameter(torch.zeros(out_features, r))
        self.B = nn.Parameter(torch.zeros(r, in_features))
        nn.init.normal_(self.B, std=1e-3)

        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        delta = (self.A @ self.B) * self.scaling
        return nn.functional.linear(x, self.weight + delta, self.bias)

class LoRAAdapter(nn.Module):
    def __init__(self, dim, r=2, alpha=1.0):
        super().__init__()
        self.lora = LoRALinear(dim, dim, r=r, alpha=alpha, init_frozen="zeros")
    def forward(self, x):
        return x + self.lora(x)

# -----------------------------
# 6. LoRA-enhanced RNN model
# -----------------------------
class SmallRNN_MoreLoRA(nn.Module):
    def __init__(self, vocab_size, embed_dim=50, hidden_dim=64, output_dim=num_classes, r=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_index)
        for p in self.embedding.parameters():
            p.requires_grad = False

        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        for p in self.rnn.parameters():
            p.requires_grad = False

        self.input_adapter  = LoRAAdapter(embed_dim, r=r, alpha=1.0)
        self.hidden_adapter = LoRAAdapter(hidden_dim, r=r, alpha=1.0)
        self.fc = LoRALinear(hidden_dim, output_dim, r=r, alpha=1.0, init_frozen="zeros")

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.input_adapter(emb)
        _, h = self.rnn(emb)
        h = h[-1]
        h = self.hidden_adapter(h)
        logits = self.fc(h)
        return logits

# -----------------------------
# 7. Instantiate model
# -----------------------------
device = torch.device("cpu")
model = SmallRNN_MoreLoRA(vocab_size=vocab_size, embed_dim=50, hidden_dim=64, r=2).to(device)

# -----------------------------
# 8. Training
# -----------------------------
criterion = nn.CrossEntropyLoss()
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=0.01)

num_epochs = 3
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

elapsed_time = time.time() - start_time
print(f"\nTotal training time: {elapsed_time:.2f} seconds")

# -----------------------------
# 9. Evaluation
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
#Total training time: 45.65 seconds
#Test Accuracy: 0.2549
