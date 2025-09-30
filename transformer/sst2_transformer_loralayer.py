import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import re, time

from loralayer import Linear, Embedding  # your LoRA layers

# -------------------------------
# 1. Load SST-2 dataset
# -------------------------------
dataset = load_dataset("glue", "sst2")
train_data = dataset["train"]
test_data = dataset["validation"]  # use validation as test

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
# 6. LoRA Transformer model
# -------------------------------
class LoRATransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, hidden_dim=256, num_layers=2, num_classes=2,
                 r=2, alpha=8, dropout=0.1, pad_idx=1):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim, r=r, lora_alpha=alpha, merge_weights=False)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            ff_linear1 = Linear(embed_dim, hidden_dim, r=r, lora_alpha=alpha, lora_dropout=dropout, merge_weights=False)
            ff_linear2 = Linear(hidden_dim, embed_dim, r=r, lora_alpha=alpha, lora_dropout=dropout, merge_weights=False)
            attn_in_proj_q = Linear(embed_dim, embed_dim, r=r, lora_alpha=alpha, lora_dropout=dropout, merge_weights=False)
            attn_in_proj_k = Linear(embed_dim, embed_dim, r=r, lora_alpha=alpha, lora_dropout=dropout, merge_weights=False)
            attn_in_proj_v = Linear(embed_dim, embed_dim, r=r, lora_alpha=alpha, lora_dropout=dropout, merge_weights=False)
            attn_out_proj = Linear(embed_dim, embed_dim, r=r, lora_alpha=alpha, lora_dropout=dropout, merge_weights=False)

            self.layers.append(nn.ModuleDict({
                "attn_q": attn_in_proj_q,
                "attn_k": attn_in_proj_k,
                "attn_v": attn_in_proj_v,
                "attn_out": attn_out_proj,
                "ff1": ff_linear1,
                "ff2": ff_linear2,
                "norm1": nn.LayerNorm(embed_dim),
                "norm2": nn.LayerNorm(embed_dim)
            }))

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.fc = Linear(embed_dim, num_classes, r=r, lora_alpha=alpha, lora_dropout=dropout, merge_weights=False)

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        seq_len, batch_size, embed_dim = x.size(1), x.size(0), x.size(2)
        x = x.permute(1,0,2)  # [seq_len, batch, embed_dim]

        for layer in self.layers:
            Q = layer["attn_q"](x)
            K = layer["attn_k"](x)
            V = layer["attn_v"](x)
            Q = Q.view(Q.size(0), Q.size(1), self.num_heads, embed_dim // self.num_heads).transpose(1,2)
            K = K.view(K.size(0), K.size(1), self.num_heads, embed_dim // self.num_heads).transpose(1,2)
            V = V.view(V.size(0), V.size(1), self.num_heads, embed_dim // self.num_heads).transpose(1,2)
            attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / (embed_dim // self.num_heads) ** 0.5
            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_probs, V)
            attn_output = attn_output.transpose(1,2).contiguous().view(seq_len, batch_size, embed_dim)
            attn_output = layer["attn_out"](attn_output)
            x = layer["norm1"](x + attn_output)
            ff_output = layer["ff2"](torch.relu(layer["ff1"](x)))
            x = layer["norm2"](x + ff_output)

        x = x.mean(dim=0)
        x = self.fc(x)
        return x

# -------------------------------
# 7. Training & Evaluation
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LoRATransformerClassifier(len(vocab)).to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params} / {total_params} ({100*trainable_params/total_params:.2f}%)")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

start = time.time()
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(texts)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
end = time.time()
print(f"Total training time: {end-start:.2f} sec")

model.eval()
total_correct, total_samples = 0,0
with torch.no_grad():
    for texts, labels in test_loader:
        texts, labels = texts.to(device), labels.to(device)
        preds = model(texts)
        total_correct += (preds.argmax(1)==labels).sum().item()
        total_samples += labels.size(0)
print(f"Test Accuracy: {100*total_correct/total_samples:.2f}%")
#training time: 31.68s
#accuracy: 0.8085