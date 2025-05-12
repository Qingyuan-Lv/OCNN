import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import CrossEntropyLoss
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

torch.set_printoptions(precision=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Gewu(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, stride=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2, bias=False),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 10,bias=False)
        )

    def forward(self, x):
        return self.model(x)

model = Gewu().to(device)

transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST("Dataset", train=True, transform=transform, download=True)

train_dataloader = DataLoader(train_dataset, 128, shuffle=True)
test_dataset = datasets.MNIST("Dataset", train=False, transform=transform, download=True)
test_dataloader = DataLoader(test_dataset, 128, shuffle=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0006, betas=(0.9, 0.999), eps=1e-8)
loss_func = CrossEntropyLoss()


for epoch in range(100):
    model.train()
    print(f"epoch: {epoch}")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            w_min, w_max = w.min().item(), w.max().item()
            print(f"{name} 权重范围: min={w_min:.6f}, max={w_max:.6f}")

    for imgs, targets in train_dataloader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = loss_func(output, targets)
        loss.backward()
        optimizer.step()

    if ((epoch%10 == 0) or (epoch == 64)):
        with torch.no_grad():
            model.eval()
            correct = 0
            for imgs, targets in test_dataloader:
                imgs, targets = imgs.to(device), targets.to(device)
                output = model(imgs)
                _, predicted = torch.max(output, dim=1)
                correct += (predicted == targets).sum().item()
            acc = correct / len(test_dataset)
            print(f"准确率: {acc * 100:.2f}%")