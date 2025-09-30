import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time


# -------------------------------
# CNN for CIFAR-10
# -------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)   # 输入通道改为3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)   # CIFAR-10 图片是 32x32，池化两次后是 8x8
        self.fc2 = nn.Linear(128, 10)           # CIFAR-10 也是10类

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # 改为 64*8*8
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -------------------------------
# 数据加载 CIFAR-10
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 三通道归一化
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


# -------------------------------
# 模型 & 训练设置
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# 打印可训练参数数量
trainable_params, total_params = 0, 0
for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        print(f"Trainable: {name} | Shape: {param.shape} | Num: {param.numel()}")
print(f"\nTotal params: {total_params}")
print(f"Trainable params: {trainable_params}")
print(f"Percentage: {100*trainable_params/total_params:.2f}%\n")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


# -------------------------------
# 训练
# -------------------------------
start = time.time()
for epoch in range(10):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 200 == 199:
            print(f"[{epoch+1}, {i+1}] loss: {running_loss/200:.3f}")
            running_loss = 0.0
end = time.time()
print(f"\nTraining finished in {end-start:.2f} sec")


# -------------------------------
# 测试集评估
# -------------------------------
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

acc = evaluate(model, testloader)
print(f"Test Accuracy: {acc:.2f}%")
#training time:77.76s
#accuracy: 0.7188