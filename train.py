# run_federated.py
import random
import torch
from transformer_model import Net
from data import get_mnist_data
from local_train import ClientTrainer
from fedavg import fedavg

def evaluate(model, loader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            elif isinstance(batch, dict):     #
                x, y = batch["x"], batch["y"]

            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total  += y.numel()
    return correct / total if total > 0 else 0.0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load data
    clients_train, test_loader = get_mnist_data(batch_size=64, root="./data")

    # load model
    global_model = Net().to(device)
    global_state = global_model.state_dict()

    trainer = ClientTrainer(model_local=lambda: Net(), device=device, lr=0.01, epochs=2)

    ROUNDS = 20

    for r in range(ROUNDS):
        nums = range(len(clients_train))
        updates = []
        for cid in nums:
            sd, n = trainer.local_train(global_state, clients_train[cid])
            updates.append((sd, n))

        # FedAvg
        global_state = fedavg(updates)
        global_model.load_state_dict(global_state, strict=True)
        global_model.to(device)

        # evaluate
        acc = evaluate(global_model, test_loader, device=device)
        print(f"[Round {r+1}/{ROUNDS}] test acc = {acc:.4f}")

if __name__ == "__main__":
    main()
