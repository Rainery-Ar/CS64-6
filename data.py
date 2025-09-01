# data.py
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
def get_mnist_data(batch_size=64, root="./data"):
    """
    return clients_train, clients_test
    clients_train[i] = DataLoader(train: samples label = i)
    """
    # normalise MNIST's [0 255] to [0,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # https://github.com/pytorch/examples/blob/main/mnist/main.py
    ])

    # download
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    clients_train = []

    for label in range(10):
        # divide labels
        targets = train_dataset.targets
        if hasattr(targets, "tolist"):
            targets = targets.tolist()
        idx_train = [i for i, y in enumerate(targets) if y == label]

        # sub-datesets
        subset_train = Subset(train_dataset, idx_train)

        # DataLoader
        loader_train = DataLoader(subset_train, batch_size=batch_size, shuffle=True)

        clients_train.append(loader_train)

    return clients_train, test_loader
