# local_trainer.py
import torch
from torch import nn, optim

class ClientTrainer:
    def __init__(self, model_local, device="cpu", lr=1e-3, epochs=2):
        """
        model: create a model object to receive parameters
        """
        self.model_local = model_local
        self.device = device
        self.lr = lr
        self.epochs = epochs

    def local_train(self, global_state_dict, train_loader):
        """return state_dict and samples n"""
        model = self.model_local().to(self.device)
        model.load_state_dict(global_state_dict)

        optimizer = optim.SGD(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(self.epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

        n_samples = len(train_loader.dataset)
        return model.state_dict(), n_samples
