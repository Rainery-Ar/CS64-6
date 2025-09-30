import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import re, time

# -------------------------------
# 1. Load SST-2 dataset
# -------------------------------
dataset = load_dataset("glue", "sst2")
train_data = dataset["train"]
test_data = dataset["validation"]  # SST-2 validation is used as test

# -------------------------------
# 2. Basic tokenizer
# -------------------------------
def basic_tokenizer(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip().split()

# -------------------------------
# 3. Build vocabulary
# -------------------------------
def build_vocab(texts, max_size=20000, min_freq=2):
    counter = Counter()
    for t in texts:
        counter.update(basic_tokenizer(t))
    vocab = {"<unk>": 0, "<pad>": 1}
    for word, freq in counter.most_common(max_size):
        if freq < min_freq: break
        vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(train_data["sentence"])
pad_idx = vocab["<pad>"]

def encode(text):
    return torch.tensor([vocab.get(tok, vocab["<unk>"]) for tok in basic_tokenizer(text)], dtype=torch.long)

# -------------------------------
# 4. Collate function for DataLoader
# -------------------------------
def collate_batch(batch):
    texts, labels = [], []
    for example in batch:
        texts.append(encode(example["sentence"]))
        labels.append(example["label"])
    texts = pad_sequence(texts, batch_first=True, padding_value=pad_idx)
    labels = torch.tensor(labels, dtype=torch.long)
    return texts, labels

# -------------------------------
# 5. DataLoader
# -------------------------------
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# -------------------------------
# 6. Transformer model
# -------------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, hidden_dim=256, num_layers=2, num_classes=2, pad_idx=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, text):
        embedded = self.embedding(text)         # [batch, seq_len, embed_dim]
        embedded = embedded.permute(1, 0, 2)    # [seq_len, batch, embed_dim]
        encoded = self.transformer(embedded)    # [seq_len, batch, embed_dim]
        pooled = encoded.mean(dim=0)            # mean pooling
        return self.fc(pooled)

# -------------------------------
# 7. Training & Evaluation
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(len(vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_model(model, loader, optimizer, criterion, epochs=3):
    start = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(texts)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (preds.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)
        print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.3f}, Acc {total_correct/total_samples:.4f}")
    end = time.time()
    print(f"\nTotal training time: {end-start:.2f} sec")

def evaluate(model, loader):
    model.eval()
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            preds = model(texts)
            total_correct += (preds.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)
    return 100 * total_correct / total_samples

# -------------------------------
# 8. Run
# -------------------------------
train_model(model, train_loader, optimizer, criterion, epochs=3)
acc = evaluate(model, test_loader)
print(f"Test Accuracy: {acc:.2f}%")
#training time:18.20s
#accuracy: 0.8085