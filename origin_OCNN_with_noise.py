import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import CrossEntropyLoss
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


torch.set_printoptions(precision=8)

data_list = [-17.95903557,-18.4629,-18.88464597,-19.3864771,-19.7049005,-19.96222363,
             -20.15809873,-20.50920363,-20.77242937,-21.11202037,-21.50936383,-21.83893493,
             -22.1017496,-22.35485607,-22.63446413,-22.94062323,-23.3072658,-23.60719043,
             -23.84921057,-24.06412503,-24.30417813,-24.55472827,-24.73722793,-24.96267927,
             -25.1840986,-25.3593453,-25.58050417,-25.7864084,-25.98559227,-26.14590267,
             -26.27316647,-26.37521157,-26.53549017,-26.68602943,-26.834303,-26.9656035,
             -27.17751977,-27.387275,-27.66765597,-28.09291057,-28.52474367,-28.9528312,
             -29.49299833,-29.98021667,-30.48903591,-30.96015911,-31.38958965,-31.87028765,
             -32.35976029,-32.79834408,-33.4642059,-33.83446223,-34.06268943,-34.25377577,
             -34.41277783,-34.71022417,-34.9680303,-35.31964819,-35.77495967,-36.10195518,
             -36.55632136,-36.97911756,-37.44859169,-37.94030805,-38.38973147,-38.82020336,
             -39.28941458,-39.76965386,-40.20999138,-40.66913494,-41.01907618,-41.4457174,
             -42.08320227,-42.55983408,-42.9903076,-43.36457103,-43.81933807,-44.4086557,
             -44.88028876,-45.32032809,-45.76617942,-46.12596521,-46.4424968,-46.7242826,
             -47.06750109,-47.49935472,-47.9691699,-48.2344309,-48.8432435,-49.36446674,
             -49.61347327,-49.9602766,-50.3301145]
data_tensor = torch.tensor(data_list, dtype=torch.float32)
min_val = data_tensor.min()
max_val = data_tensor.max()
normalized_tensor = 1.2 * ((data_tensor - min_val) / (max_val - min_val)) - 0.6

discrete_values = normalized_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class STEQuantizerWithNoise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, discrete_values, sigma=0.3333):
        w_flat = w.view(-1)
        index = torch.argmin(torch.abs(w_flat[:, None] - discrete_values[None, :]), dim=1)
        quantized = discrete_values[index].view_as(w)

        noise = torch.randn_like(quantized) * sigma * torch.abs(quantized)
        noisy_quantized = quantized + noise

        return noisy_quantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class QuantizedConv2d(nn.Conv2d):
    def __init__(self, *args, discrete_values, sigma=0.3333, loss_factor=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.discrete_values = discrete_values.to(device)
        self.sigma = sigma
        self.loss_factor = loss_factor

    def loss_function(self,output):
        return 1 / (1 + 1 * self.loss_factor * torch.abs(output) ** 2)

    def forward(self, x):
        quantized_weight = STEQuantizerWithNoise.apply(self.weight, self.discrete_values, self.sigma)
        output =  (F.conv2d(x, quantized_weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups))
        loss_map = self.loss_function(output)
        output = output * loss_map
        return  output


class QuantizedLinear(nn.Linear):
    def __init__(self, *args, discrete_values, sigma=0.3333, loss_factor=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.discrete_values = discrete_values.to(device)
        self.sigma = sigma
        self.loss_factor = loss_factor

    def loss_function(self,output):
        return 1 / (1 + 1 * self.loss_factor * torch.abs(output) ** 2)

    def forward(self, x):
        quantized_weight = STEQuantizerWithNoise.apply(self.weight, self.discrete_values, self.sigma)
        output = F.linear(x, quantized_weight, self.bias)
        loss_map = self.loss_function(output)
        output = output * loss_map
        return  output

class Gewu(nn.Module):
    def __init__(self,sigma=0.3333,loss_factor=0.5):
        super().__init__()
        self.model = nn.Sequential(
            QuantizedConv2d(1, 32, 3, padding=1, stride=2, bias=True, discrete_values=discrete_values,sigma=sigma,loss_factor=loss_factor),
            nn.ReLU(),
            QuantizedConv2d(32, 64, 3, padding=1, stride=2, bias=True, discrete_values=discrete_values,sigma=sigma,loss_factor=loss_factor),
            nn.ReLU(),
            QuantizedConv2d(64, 128, 3, padding=1, stride=2, bias=True, discrete_values=discrete_values,sigma=sigma,loss_factor=loss_factor),
            nn.ReLU(),
            nn.Flatten(),
            QuantizedLinear(128 * 4 * 4, 10, bias=True, discrete_values=discrete_values,sigma=sigma,loss_factor=loss_factor)
        )

    def forward(self, x):
        return self.model(x)


model = Gewu(sigma=2.0,loss_factor=2.0).to(device)

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


for epoch in range(150):
    model.train()
    print(f"epoch: {epoch}")
    for imgs, targets in train_dataloader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = loss_func(output, targets)
        loss.backward()
        optimizer.step()

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