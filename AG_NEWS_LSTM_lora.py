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

train_labels = [label for label, _ in train_list]
min_label, max_label = min(train_labels), max(train_labels)
num_classes = max_label + 1 if min_label == 0 else max_label
pad_index = 0

all_tokens = torch.cat([text for _, text in train_list])
vocab_size = int(all_tokens.max()) + 1

def collate_batch(batch):
    texts, labels = [], []
    for label, text in batch:
        texts.append(text.clone())
        mapped_label = label - 1 if min_label == 1 else label
        labels.append(mapped_label)
    texts = pad_sequence(texts, batch_first=True, padding_value=pad_index)
    labels = torch.tensor(labels, dtype=torch.long)
    return texts, labels

batch_size = 32
train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# -----------------------------
# 2. LoRA Linear
# -----------------------------
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=2, alpha=1.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)

    def forward(self, x):
        base = nn.functional.linear(x, self.weight, self.bias)
        update = nn.functional.linear(x, self.B @ self.A)
        return base + self.alpha * update

# -----------------------------
# 3. LoRA Embedding
# -----------------------------
class LoRAEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, r=2, alpha=1.0, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.embedding.weight.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, embedding_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(num_embeddings, r) * 0.01)
        self.alpha = alpha

    def forward(self, x):
        return self.embedding(x) + self.alpha * (self.B[x] @ self.A)

# -----------------------------
# 4. LoRA LSTM wrapper
# -----------------------------
class LoRALSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True, r=2):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=bidirectional)

        # Replace LSTM weight_ih and weight_hh with LoRA adapters
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                param.requires_grad = False
                rnk = r
                # Create LoRA adapters
                setattr(self, name + '_A', nn.Parameter(torch.randn(rnk, param.size(1)) * 0.01))
                setattr(self, name + '_B', nn.Parameter(torch.randn(param.size(0), rnk) * 0.01))

    def forward(self, x):
        # Compute LoRA updates for weight_ih and weight_hh
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                A = getattr(self, name + '_A')
                B = getattr(self, name + '_B')
                param.data += B @ A  # inject LoRA update
        output, (hidden, cell) = self.lstm(x)
        return output, (hidden, cell)

# -----------------------------
# 5. Full BiLSTM model with LoRA everywhere
# -----------------------------
class SmallBiLSTM_FullLoRA(nn.Module):
    def __init__(self, vocab_size, embed_dim=50, hidden_dim=64, output_dim=num_classes, dropout=0.3, r=2):
        super().__init__()
        self.embedding = LoRAEmbedding(vocab_size, embed_dim, r=r, padding_idx=pad_index)
        self.lstm = LoRALSTM(embed_dim, hidden_dim, bidirectional=True, r=r)
        self.dropout = nn.Dropout(dropout)
        self.fc = LoRALinear(hidden_dim*2, output_dim, r=r)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden_cat = self.dropout(hidden_cat)
        return self.fc(hidden_cat)

# -----------------------------
# 6. Training
# -----------------------------
device = torch.device("cpu")
model = SmallBiLSTM_FullLoRA(vocab_size=vocab_size, r=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
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
#Total training time: 89.62 seconds
#Test Accuracy: 0.6414