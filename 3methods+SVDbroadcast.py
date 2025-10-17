# -*- coding: utf-8 -*-
# Federated LoRA on MNIST — Optimized Zero-Pad / Rank-Based / SVD
# + SVD Broadcast (wrapper) that can stack with inner aggregators
# Optimizations: Smart padding, rank selection, adaptive learning rates, momentum, quality weighting

# 如果你只想跑某个组合，把 agg_list 调整为你想要的项即可。
# target_rank 控制共享子空间维度（也是全局 LoRA 的 rank）。
# svd_broadcast_shared 会在轮次之间自动维持“共享子空间”的连续性，即“广播”的效果；无须额外改动客户端训练。

import torch, random, copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
CFG = dict(
    seed=42,
    device="cuda" if torch.cuda.is_available() else "cpu",

    # Clients & data
    num_clients=10,
    classes_per_client=2,          # Non-IID: labels per client
    batch_size=256,
    max_items_per_client=7000,     # cap per-client samples for runtime

    # Non-IID & "double-imbalance" (probabilistic)
    double_imbalance=True,

    # Training (NO pretrain)
    base_pretrain_epochs=0,        # keep 0 per teacher
    base_lr=1e-3,
    local_epochs=2,
    lr_local=2e-3,
    weight_decay=1e-4,
    global_rounds=35,

    # LoRA
    alpha=16,
    client_ranks=[2,4,8,4,2,8,4,2,8,4],   # ranks for the 10 clients
    target_rank=8,                        # global/aggregation target rank

    # Rank-based settings - OPTIMIZED: Choose higher rank
    rank_based_expect_rank=8,  # Changed from 4 to 8 for better capacity

    # NO finetune (per your request)
    do_finetune=False,
    finetune_global_epochs=4,
    finetune_lr=1.5e-3,

    # Visualization
    figsize=(8,4.5),

    # OPTIMIZATION PARAMETERS
    use_momentum=True,           # Add momentum to aggregation
    momentum_beta=0.9,           # Momentum coefficient
    adaptive_lr=True,            # Use adaptive learning rates
    smart_padding=True,          # Intelligent padding strategy
    quality_weighting=True,      # Weight by both size and quality
)

# =========================
# Utils
# =========================
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed(CFG["seed"])
DEVICE = torch.device(CFG["device"])

# =========================
# Model & LoRA
# =========================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.drop = nn.Dropout(0.1)
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x)); x = self.drop(x)
        x = F.relu(self.fc2(x)); x = self.drop(x)
        return self.fc3(x)

class LoRALinear(nn.Module):
    """
    Freeze W0; train ΔW = B @ A; forward: W0 x + (alpha/r) * B A x
    Init per LoRA paper: A ~ N(0,·), B = 0
    """
    def __init__(self, base_linear: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.W0 = base_linear
        for p in self.W0.parameters():
            p.requires_grad = False
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.randn(rank, self.W0.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.W0.out_features, rank))
    def forward(self, x):
        out = self.W0(x)
        delta = (x @ self.lora_A.t()) @ self.lora_B.t()
        return out + (self.alpha / self.rank) * delta

def wrap_with_lora(model: nn.Module, rank: int, alpha: int):
    m = copy.deepcopy(model)
    m.fc1 = LoRALinear(m.fc1, rank, alpha)
    m.fc2 = LoRALinear(m.fc2, rank, alpha)
    m.fc3 = LoRALinear(m.fc3, rank, alpha)
    return m

def get_lora_state(model: nn.Module):
    state = {}
    for name in ["fc1","fc2","fc3"]:
        L: LoRALinear = getattr(model, name)
        state[name] = {
            "A": L.lora_A.detach().cpu().clone(),
            "B": L.lora_B.detach().cpu().clone(),
            "in": L.W0.in_features, "out": L.W0.out_features, "rank": L.rank, "alpha": L.alpha
        }
    return state

# Broadcast global -> client (truncate/pad to client's rank; no new layer)
def broadcast_global_to_client(client_model: nn.Module, global_state):
    for name in ["fc1","fc2","fc3"]:
        Lc: LoRALinear = getattr(client_model, name)
        A_g = global_state[name]["A"]; B_g = global_state[name]["B"]
        r_g = A_g.shape[0]; r_c = Lc.rank
        in_ = Lc.W0.in_features; out_ = Lc.W0.out_features
        A_new = torch.zeros((r_c, in_))
        B_new = torch.zeros((out_, r_c))
        r_min = min(r_c, r_g)
        A_new[:r_min, :] = A_g[:r_min, :]
        B_new[:, :r_min] = B_g[:, :r_min]
        with torch.no_grad():
            Lc.lora_A.copy_(A_new.to(Lc.lora_A.device))
            Lc.lora_B.copy_(B_new.to(Lc.lora_B.device))

# After aggregation, load into global (global rank fixed = target_rank)
def load_agg_into_global(global_model: nn.Module, agg_state, target_rank:int):
    for name in ["fc1","fc2","fc3"]:
        Lg: LoRALinear = getattr(global_model, name)
        assert Lg.rank == target_rank
        A_ag = agg_state[name]["A"]; B_ag = agg_state[name]["B"]
        r_ag = A_ag.shape[0]; in_ = Lg.W0.in_features; out_ = Lg.W0.out_features
        A_new = torch.zeros((target_rank, in_)); B_new = torch.zeros((out_, target_rank))
        r_min = min(target_rank, r_ag)
        A_new[:r_min, :] = A_ag[:r_min, :]
        B_new[:, :r_min] = B_ag[:, :r_min]
        with torch.no_grad():
            Lg.lora_A.copy_(A_new.to(Lg.lora_A.device))
            Lg.lora_B.copy_(B_new.to(Lg.lora_B.device))

# =========================
# Data
# =========================
def build_mnist_loaders():
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    return train_ds, test_ds

def indices_by_label_fast(ds):
    if hasattr(ds, "targets"): targets = ds.targets
    elif hasattr(ds, "train_labels"): targets = ds.train_labels
    else: targets = [ds[i][1] for i in range(len(ds))]
    if torch.is_tensor(targets): targets = targets.tolist()
    buckets = defaultdict(list)
    for i, y in enumerate(targets): buckets[int(y)].append(i)
    for k in buckets: random.shuffle(buckets[k])
    return buckets

def make_partitions_double_imbalance(train_ds, num_clients, classes_per_client, max_items_per_client):
    lab2idx = indices_by_label_fast(train_ds)
    probs = np.array([num_clients - i for i in range(10)], dtype=np.float64); probs /= probs.sum()

    client_labels = []
    for _ in range(num_clients):
        labs = np.random.choice(np.arange(10), size=classes_per_client, replace=False, p=probs)
        client_labels.append(sorted(list(map(int, labs))))

    total_train = len(train_ds)
    weights = np.linspace(1.4, 0.6, num_clients); weights /= weights.sum()

    client_indices = []
    for c in range(num_clients):
        pool = []
        for lab in client_labels[c]: pool.extend(lab2idx[lab])
        random.shuffle(pool)
        quota = min(int(total_train * weights[c]), max_items_per_client)
        client_indices.append(pool[:quota])

    return client_indices, client_labels

def make_loader_from_indices(ds, idxs, batch_size, shuffle=True):
    return DataLoader(Subset(ds, idxs), batch_size=batch_size, shuffle=shuffle, drop_last=False)

# =========================
# Train / Test / Quality Assessment
# =========================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = ce(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

    return total_loss / total_samples  # Return average loss for quality assessment

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot=0; corr=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        tot += y.size(0)
        corr += (pred==y).sum().item()
    return 100.0*corr/tot

@torch.no_grad()
def evaluate_quality(model, loader, device):
    """Evaluate both accuracy and loss for quality weighting"""
    model.eval()
    tot=0; corr=0; total_loss=0.0
    ce = nn.CrossEntropyLoss()
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(1)
        loss = ce(out, y)
        tot += y.size(0)
        corr += (pred==y).sum().item()
        total_loss += loss.item() * y.size(0)

    acc = 100.0*corr/tot
    avg_loss = total_loss/tot
    return acc, avg_loss

# =========================
# OPTIMIZED Aggregators (existing)
# =========================
def zero_pad_product_sum_optimized(states, weights, target_rank, prev_global_state=None):
    """
    OPTIMIZATION 1: Smart Padding Strategy
    Instead of zero-padding, use weighted interpolation or truncation
    """
    print(f"Zero Pad (Optimized) - Using {len(states)} clients with weights: {[f'{w:.4f}' for w in weights]}")

    agg = {}
    for layer in ["fc1","fc2","fc3"]:
        out_ = states[0][layer]["out"]; in_ = states[0][layer]["in"]
        A_sum = torch.zeros((target_rank, in_)); B_sum = torch.zeros((out_, target_rank))

        for st, w in zip(states, weights):
            A = st[layer]["A"]; B = st[layer]["B"]; r = st[layer]["rank"]

            if CFG["smart_padding"] and r < target_rank:
                # SMART PADDING: small random fill instead of zeros
                A_pad = torch.zeros((target_rank, in_))
                B_pad = torch.zeros((out_, target_rank))
                A_pad[:r,:] = A
                B_pad[:,:r] = B
                if r < target_rank:
                    remaining_A = torch.randn(target_rank - r, in_) * 0.001
                    remaining_B = torch.randn(out_, target_rank - r) * 0.001
                    A_pad[r:,:] = remaining_A
                    B_pad[:,r:] = remaining_B
            else:
                # Standard padding
                A_pad = torch.zeros((target_rank, in_)); A_pad[:r,:] = A
                B_pad = torch.zeros((out_, target_rank));  B_pad[:,:r] = B

            A_sum += w * A_pad; B_sum += w * B_pad

        agg[layer] = {"A": A_sum, "B": B_sum}

    return agg

def rank_based_product_sum_optimized(states, weights, expect_rank=8):
    """
    OPTIMIZATION 2: Use higher rank (8 instead of 4) for better capacity
    """
    ranks = set(st["fc1"]["rank"] for st in states)
    assert len(ranks)==1 and list(ranks)[0]==expect_rank, f"need rank={expect_rank}, got {ranks}"

    print(f"Rank-Based (Optimized) - Using {len(states)} clients (rank={expect_rank}) with weights: {[f'{w:.4f}' for w in weights]}")

    agg = {}
    for layer in ["fc1","fc2","fc3"]:
        in_ = states[0][layer]["in"]; out_ = states[0][layer]["out"]; r = expect_rank
        A_sum = torch.zeros((r, in_)); B_sum = torch.zeros((out_, r))
        for st,w in zip(states,weights):
            A_sum += w * st[layer]["A"]; B_sum += w * st[layer]["B"]
        agg[layer] = {"A": A_sum, "B": B_sum}
    return agg

def svd_sum_product(states, weights, target_rank):
    """Keep SVD unchanged for fair comparison"""
    print(f"SVD - Using {len(states)} clients with weights: {[f'{w:.4f}' for w in weights]}")

    agg = {}
    for layer in ["fc1","fc2","fc3"]:
        in_ = states[0][layer]["in"]; out_ = states[0][layer]["out"]
        DW_sum = torch.zeros((out_, in_))
        for st, w in zip(states, weights):
            DW_sum += w * (st[layer]["B"] @ st[layer]["A"])
        U, S, Vh = torch.linalg.svd(DW_sum, full_matrices=False)
        r = min(target_rank, S.numel())
        U = U[:, :r]; Vh = Vh[:r, :]; S = S[:r]
        sqrtS = torch.sqrt(S + 1e-8)
        B_new = U * sqrtS.unsqueeze(0)          # out x r
        A_new = sqrtS.unsqueeze(1) * Vh         # r x in
        agg[layer] = {"A": A_new, "B": B_new}
    return agg

# =========================
# NEW: SVD Broadcast (wrapper) — can stack with inner aggregators
# =========================
def svd_broadcast_aggregate(states, weights, target_rank, prev_shared_subspace=None, inner="svd"):
    """
    SVD Broadcast (外壳)：
      1) 估计/更新每层共享子空间 (U_shared, V_shared) ∈ R^{out×r}, R^{r×in}
      2) 将每个客户端 ΔW_i = B_i A_i 投影到子空间：M_i = U_s^T ΔW_i V_s^T ∈ R^{r×r}
      3) 在 r×r 子空间内用 'inner' 聚合器（svd / rank_based / zero_pad）做加权聚合
      4) 用共享子空间把子空间内的聚合结果“抬回”原空间，得到全局 A,B

    返回：
      agg_state: {layer: {"A": A_new, "B": B_new}}
      shared_subspace: {layer: {"U": U_s, "V": V_s}}  # 用于下一轮 warm-start
    """
    assert inner in ("svd", "rank_based", "zero_pad")
    print(f"SVD Broadcast (inner={inner}) - Using {len(states)} clients with weights: {[f'{w:.4f}' for w in weights]}")

    agg = {}
    shared_subspace = {}

    for layer in ["fc1","fc2","fc3"]:
        in_  = states[0][layer]["in"]
        out_ = states[0][layer]["out"]

        # 1) 共享子空间：首次用 DW 的加权和初始化，后续 warm-start
        if prev_shared_subspace is not None and layer in prev_shared_subspace:
            U_s = prev_shared_subspace[layer]["U"].clone()
            V_s = prev_shared_subspace[layer]["V"].clone()
        else:
            DW_sum = torch.zeros((out_, in_))
            for st, w in zip(states, weights):
                DW_sum += w * (st[layer]["B"] @ st[layer]["A"])
            U, S, Vh = torch.linalg.svd(DW_sum, full_matrices=False)
            r_eff = min(target_rank, S.numel())
            U_s = U[:, :r_eff]         # (out x r)
            V_s = Vh[:r_eff, :]        # (r x in)
            # 如果奇异值不足 target_rank，做安全补齐（极少见，保持形状一致）
            if r_eff < target_rank:
                # 简单补齐：随机正交方向（此处用零填充以简化；可替换为Householder扩展）
                U_pad = torch.zeros((out_, target_rank - r_eff))
                V_pad = torch.zeros((target_rank - r_eff, in_))
                U_s = torch.cat([U_s, U_pad], dim=1)
                V_s = torch.cat([V_s, V_pad], dim=0)

        r = target_rank
        # 2) 投影到子空间：M_i = U_s^T ΔW_i V_s^T (r x r)
        mids = []
        for st in states:
            DW = st[layer]["B"] @ st[layer]["A"]       # (out x in)
            M  = U_s.T @ DW @ V_s.T                    # (r x r)
            # 将 M_i 转回 LoRA 形式（为了统一内聚合接口）
            Um, Sm, VmT = torch.linalg.svd(M, full_matrices=False)
            sqrtS = torch.sqrt(Sm + 1e-8)
            B_mid = Um * sqrtS.unsqueeze(0)            # (r x r)
            A_mid = (sqrtS.unsqueeze(1) * VmT)         # (r x r)
            mids.append({"A": A_mid, "B": B_mid, "in": r, "out": r, "rank": r, "alpha": 1})

        # 3) 子空间内的“内聚合器”
        if inner == "svd":
            DW_sum_mid = torch.zeros((r, r))
            for mid, w in zip(mids, weights):
                DW_sum_mid += w * (mid["B"] @ mid["A"])        # r x r
            U_m, S_m, Vh_m = torch.linalg.svd(DW_sum_mid, full_matrices=False)
            r_m = min(r, S_m.numel())
            U_m = U_m[:, :r_m]; Vh_m = Vh_m[:r_m, :]; S_m = S_m[:r_m]
            sqrtSm = torch.sqrt(S_m + 1e-8)
            B_in = U_m * sqrtSm.unsqueeze(0)                   # (r x r_m)
            A_in = sqrtSm.unsqueeze(1) * Vh_m                  # (r_m x r)

            # 如果 r_m < r，安全补齐（零填充）
            if r_m < r:
                B_pad = torch.zeros((r, r - r_m)); A_pad = torch.zeros((r - r_m, r))
                B_in = torch.cat([B_in, B_pad], dim=1)         # (r x r)
                A_in = torch.cat([A_in, A_pad], dim=0)         # (r x r)

        elif inner in ("rank_based", "zero_pad"):
            # 这里 r 对所有 mid 一致，无需额外对齐，等价于加权和
            A_in = torch.zeros((r, r)); B_in = torch.zeros((r, r))
            for mid, w in zip(mids, weights):
                A_in += w * mid["A"]
                B_in += w * mid["B"]

        # 4) 从子空间“抬回”原空间：B_new = U_s @ B_in, A_new = A_in @ V_s
        B_new = U_s @ B_in                                   # (out x r)
        A_new = A_in @ V_s                                   # (r x in)
        agg[layer] = {"A": A_new, "B": B_new}
        shared_subspace[layer] = {"U": U_s, "V": V_s}

    return agg, shared_subspace

# OPTIMIZATION 3: Momentum-based aggregation
class MomentumTracker:
    def __init__(self):
        self.momentum_state = {}

    def apply_momentum(self, current_state, beta=0.9):
        if not self.momentum_state:
            self.momentum_state = copy.deepcopy(current_state)
            return current_state

        momentum_state = {}
        for layer in current_state:
            momentum_state[layer] = {}
            for key in current_state[layer]:
                if isinstance(current_state[layer][key], torch.Tensor):
                    # v = β * v_prev + (1-β) * current
                    self.momentum_state[layer][key] = (beta * self.momentum_state[layer][key] +
                                                      (1-beta) * current_state[layer][key])
                    momentum_state[layer][key] = self.momentum_state[layer][key]
                else:
                    momentum_state[layer][key] = current_state[layer][key]

        return momentum_state

# =========================
# Federated runner (single aggregator) - OPTIMIZED
# =========================
def run_federated_once(aggregator_name: str, shared_objects=None, seed_offset=0):
    """
    aggregator_name options:
      - "zero_pad_product_sum"
      - "rank_based_product_sum"
      - "svd_sum_product"
      - "svd_broadcast+svd"
      - "svd_broadcast+rank_based"
      - "svd_broadcast+zero_pad"
    """
    set_seed(CFG["seed"] + seed_offset)

    # ----- datasets (reuse if provided)
    if shared_objects is None or "train_ds" not in shared_objects:
        train_ds, test_ds = build_mnist_loaders()
        client_indices, client_labels = make_partitions_double_imbalance(
            train_ds, CFG["num_clients"], CFG["classes_per_client"], CFG["max_items_per_client"]
        )
        client_loaders = [make_loader_from_indices(train_ds, idxs, CFG["batch_size"], shuffle=True)
                          for idxs in client_indices]
        client_nsamples = [len(idxs) for idxs in client_indices]
        shared_objects = dict(train_ds=train_ds, test_ds=test_ds,
                              client_indices=client_indices, client_labels=client_labels,
                              client_nsamples=client_nsamples)
    else:
        train_ds = shared_objects["train_ds"]; test_ds = shared_objects["test_ds"]
        client_indices = shared_objects["client_indices"]; client_labels = shared_objects["client_labels"]
        client_nsamples = shared_objects["client_nsamples"]
        client_loaders = [make_loader_from_indices(train_ds, idxs, CFG["batch_size"], shuffle=True)
                          for idxs in client_indices]

    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)

    # ----- init base + global (NO pretrain)
    base = MLP().to(DEVICE)
    global_model = wrap_with_lora(base, rank=CFG["target_rank"], alpha=CFG["alpha"]).to(DEVICE)

    # initial acc (before any aggregation)
    init_acc = evaluate(global_model, test_loader, DEVICE)

    # ----- client models with their ranks
    client_models = []
    for cid in range(CFG["num_clients"]):
        m = wrap_with_lora(base, rank=CFG["client_ranks"][cid], alpha=CFG["alpha"]).to(DEVICE)
        client_models.append(m)

    # Initialize momentum tracker
    momentum_tracker = MomentumTracker() if CFG["use_momentum"] else None

    # --- NEW: keep shared subspace across rounds for SVD broadcast
    svd_broadcast_shared = None

    # ----- rounds
    curve = []
    best_no_ft = 0.0
    for rd in range(1, CFG["global_rounds"]+1):
        g_state = get_lora_state(global_model)
        for m in client_models:
            broadcast_global_to_client(m, g_state)

        returned_states = []; returned_ns = []; client_qualities = []
        for cid, m in enumerate(client_models):
            # Adaptive LR per round
            if CFG["adaptive_lr"]:
                lr = CFG["lr_local"] * (0.98 ** (rd - 1))
            else:
                lr = CFG["lr_local"]

            opt = optim.Adam([p for p in m.parameters() if p.requires_grad],
                             lr=lr, weight_decay=CFG["weight_decay"])

            total_loss = 0.0
            for _ in range(CFG["local_epochs"]):
                loss = train_one_epoch(m, client_loaders[cid], opt, DEVICE)
                total_loss += loss

            returned_states.append(get_lora_state(m))
            returned_ns.append(shared_objects["client_nsamples"][cid])

            # Quality-based weighting
            if CFG["quality_weighting"]:
                acc, avg_loss = evaluate_quality(m, client_loaders[cid], DEVICE)
                quality = acc / (1 + avg_loss)
                client_qualities.append(quality)

        # weights（样本量×质量）
        if CFG["quality_weighting"] and client_qualities:
            combined_weights = [ns * q for ns, q in zip(returned_ns, client_qualities)]
            w = np.array(combined_weights, dtype=np.float64)
        else:
            w = np.array(returned_ns, dtype=np.float64)
        w = (w / w.sum()).tolist()

        # Aggregation switch
        if aggregator_name == "zero_pad_product_sum":
            agg = zero_pad_product_sum_optimized(returned_states, w, target_rank=CFG["target_rank"])

        elif aggregator_name == "rank_based_product_sum":
            expect_rank = CFG["rank_based_expect_rank"]
            sel = [i for i in range(CFG["num_clients"]) if CFG["client_ranks"][i]==expect_rank]
            assert len(sel)>0, "no same-rank clients"
            states_sel = [returned_states[i] for i in sel]
            sizes_sel  = [returned_ns[i] for i in sel]
            if CFG["quality_weighting"] and client_qualities:
                qualities_sel = [client_qualities[i] for i in sel]
                combined_weights = [s * q for s, q in zip(sizes_sel, qualities_sel)]
                weights = [ww / sum(combined_weights) for ww in combined_weights]
            else:
                weights = [s / sum(sizes_sel) for s in sizes_sel]

            print(f"Rank-based selected clients {sel} with sample sizes {sizes_sel}")
            agg = rank_based_product_sum_optimized(states_sel, weights, expect_rank=expect_rank)

        elif aggregator_name == "svd_sum_product":
            agg = svd_sum_product(returned_states, w, target_rank=CFG["target_rank"])

        elif aggregator_name in ("svd_broadcast+svd", "svd_broadcast+rank_based", "svd_broadcast+zero_pad"):
            inner = aggregator_name.split("+", 1)[1]  # "svd" | "rank_based" | "zero_pad"
            agg, svd_broadcast_shared = svd_broadcast_aggregate(
                returned_states, w, target_rank=CFG["target_rank"],
                prev_shared_subspace=svd_broadcast_shared,
                inner=("svd" if inner=="svd" else ("rank_based" if inner=="rank_based" else "zero_pad"))
            )

        else:
            raise ValueError(f"Unknown aggregator: {aggregator_name}")

        # Momentum (optional)
        if CFG["use_momentum"] and momentum_tracker:
            agg = momentum_tracker.apply_momentum(agg, beta=CFG["momentum_beta"])

        # Load into global & eval
        load_agg_into_global(global_model, agg, target_rank=CFG["target_rank"])

        acc = evaluate(global_model, test_loader, DEVICE)
        best_no_ft = max(best_no_ft, acc)
        curve.append(acc)
        print(f"[{aggregator_name}] Round {rd:02d}  Test Acc (NO finetune) = {acc:.2f}%")

    final_no_ft = curve[-1] if len(curve)>0 else init_acc
    return {
        "init_acc": init_acc,
        "curve": curve,
        "final_no_ft": final_no_ft,
        "best_no_ft": best_no_ft,
        "shared": shared_objects
    }

# =========================
# Run all (NO FINETUNE)
# =========================
if __name__ == "__main__":
    set_seed(CFG["seed"])

    # Build one shared split for fair comparison
    train_ds, test_ds = build_mnist_loaders()
    client_indices, client_labels = make_partitions_double_imbalance(
        train_ds, CFG["num_clients"], CFG["classes_per_client"], CFG["max_items_per_client"]
    )
    client_nsamples = [len(idxs) for idxs in client_indices]
    shared = dict(train_ds=train_ds, test_ds=test_ds,
                  client_indices=client_indices, client_labels=client_labels,
                  client_nsamples=client_nsamples)

    print("=== OPTIMIZATION SETTINGS ===")
    print(f"Smart Padding: {CFG['smart_padding']}")
    print(f"Use Momentum: {CFG['use_momentum']}")
    print(f"Adaptive LR: {CFG['adaptive_lr']}")
    print(f"Quality Weighting: {CFG['quality_weighting']}")
    print(f"Rank-Based Expect Rank: {CFG['rank_based_expect_rank']}")
    print()
    print("Client label sets:", client_labels)
    print("Client sample sizes:", client_nsamples)
    print("Client ranks:", CFG["client_ranks"])

    results = {}
    agg_list = [
        # "zero_pad_product_sum",
        # "rank_based_product_sum",
        # "svd_sum_product",
        # NEW: three broadcast+inner combos
        "svd_broadcast+svd",
        "svd_broadcast+rank_based",
        "svd_broadcast+zero_pad",
    ]
    for i, agg in enumerate(agg_list):
        print(f"\n=== Running: {agg} (Optimized) ===\n")
        out = run_federated_once(agg, shared_objects=shared, seed_offset=i+1)
        results[agg] = out

    # ---------- Summary ----------
    init_acc = results[agg_list[0]]["init_acc"]  # same init for all
    print("\n====== SUMMARY (MNIST, Optimized Methods) ======")
    print(f"Initial global (before rounds) : {init_acc:.2f}%")
    for agg, out in results.items():
        print(f"[{agg}]  NO-finetune (last/best): {out['final_no_ft']:.2f}% / {out['best_no_ft']:.2f}%")

    # ---------- Visualization ----------
    # 1) Training curves (NO finetune)
    plt.figure(figsize=CFG["figsize"])
    label_map = {
        "zero_pad_product_sum": "Zero Pad (Optimized)",
        "rank_based_product_sum": "Rank-Based (Optimized)",
        "svd_sum_product": "SVD",
        "svd_broadcast+svd": "SVD Broadcast + SVD",
        "svd_broadcast+rank_based": "SVD Broadcast + Rank-Based",
        "svd_broadcast+zero_pad": "SVD Broadcast + Zero-Pad",
    }
    for agg, out in results.items():
        plt.plot(range(1, len(out["curve"])+1), out["curve"], label=label_map.get(agg, agg))
    plt.axhline(init_acc, linestyle="--", linewidth=1, label="initial")
    plt.xlabel("Global Round"); plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy per Round (Optimized & SVD Broadcast)")
    plt.legend(); plt.tight_layout()
    plt.show()

    # 2) Final comparison bars (NO finetune only)
    names = [label_map[a] for a in agg_list]
    last_noft = [results[a]["final_no_ft"] for a in agg_list]

    x = np.arange(len(names)); w = 0.55
    plt.figure(figsize=CFG["figsize"])
    bars = plt.bar(x, last_noft, width=w)
    plt.xticks(x, names, rotation=12, ha="right")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Final Accuracy Comparison (Optimized & SVD Broadcast)")

    # Add value labels on bars
    for bar, val in zip(bars, last_noft):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
