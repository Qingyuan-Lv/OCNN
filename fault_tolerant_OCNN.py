import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import math

torch.set_printoptions(precision=8)

data_list = [-17.95903557, -18.4629, -18.88464597, -19.3864771, -19.7049005, -19.96222363,
             -20.15809873, -20.50920363, -20.77242937, -21.11202037, -21.50936383, -21.83893493,
             -22.1017496, -22.35485607, -22.63446413, -22.94062323, -23.3072658, -23.60719043,
             -23.84921057, -24.06412503, -24.30417813, -24.55472827, -24.73722793, -24.96267927,
             -25.1840986, -25.3593453, -25.58050417, -25.7864084, -25.98559227, -26.14590267,
             -26.27316647, -26.37521157, -26.53549017, -26.68602943, -26.834303, -26.9656035,
             -27.17751977, -27.387275, -27.66765597, -28.09291057, -28.52474367, -28.9528312,
             -29.49299833, -29.98021667, -30.48903591, -30.96015911, -31.38958965, -31.87028765,
             -32.35976029, -32.79834408, -33.4642059, -33.83446223, -34.06268943, -34.25377577,
             -34.41277783, -34.71022417, -34.9680303, -35.31964819, -35.77495967, -36.10195518,
             -36.55632136, -36.97911756, -37.44859169, -37.94030805, -38.38973147, -38.82020336,
             -39.28941458, -39.76965386, -40.20999138, -40.66913494, -41.01907618, -41.4457174,
             -42.08320227, -42.55983408, -42.9903076, -43.36457103, -43.81933807, -44.4086557,
             -44.88028876, -45.32032809, -45.76617942, -46.12596521, -46.4424968, -46.7242826,
             -47.06750109, -47.49935472, -47.9691699, -48.2344309, -48.8432435, -49.36446674,
             -49.61347327, -49.9602766, -50.3301145]
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

    def loss_function(self, output):
        return 1 / (1 + 1 * self.loss_factor * torch.abs(output) ** 2)

    def forward(self, x):
        quantized_weight = STEQuantizerWithNoise.apply(self.weight, self.discrete_values, self.sigma)
        output = F.conv2d(x, quantized_weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        loss_map = self.loss_function(output)
        output = output * loss_map
        return output


class QuantizedLinear(nn.Linear):
    def __init__(self, *args, discrete_values, sigma=0.3333, loss_factor=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.discrete_values = discrete_values.to(device)
        self.sigma = sigma
        self.loss_factor = loss_factor

    def loss_function(self, output):
        return 1 / (1 + 1 * self.loss_factor * torch.abs(output) ** 2)

    def forward(self, x):
        quantized_weight = STEQuantizerWithNoise.apply(self.weight, self.discrete_values, self.sigma)
        output = F.linear(x, quantized_weight, self.bias)
        loss_map = self.loss_function(output)
        output = output * loss_map
        return output


class Gewu(nn.Module):
    def __init__(self, sigma=2.0, loss_factor=2.0):
        super().__init__()

        self.conv1_1 = QuantizedConv2d(1, 32, 3, padding=1, bias=True, discrete_values=discrete_values, sigma=sigma,
                                       loss_factor=loss_factor)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = QuantizedConv2d(32, 32, 3, padding=1, bias=True, discrete_values=discrete_values, sigma=sigma,
                                       loss_factor=loss_factor)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv2_1 = QuantizedConv2d(32, 64, 3, padding=1, bias=True, discrete_values=discrete_values, sigma=sigma,
                                       loss_factor=loss_factor)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = QuantizedConv2d(64, 64, 3, padding=1, bias=True, discrete_values=discrete_values, sigma=sigma,
                                       loss_factor=loss_factor)
        self.bn2_2 = nn.BatchNorm2d(64)

        self.shortcut1 = QuantizedConv2d(1, 32, 1, bias=True, discrete_values=discrete_values, sigma=sigma,
                                         loss_factor=loss_factor)
        self.shortcut2 = QuantizedConv2d(32, 64, 1, bias=True, discrete_values=discrete_values, sigma=sigma,
                                         loss_factor=loss_factor)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            QuantizedLinear(64 * 7 * 7, 256, bias=True, discrete_values=discrete_values, sigma=sigma, loss_factor=loss_factor),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            QuantizedLinear(256, 128, bias=True, discrete_values=discrete_values, sigma=sigma, loss_factor=loss_factor),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            QuantizedLinear(128, 10, bias=True, discrete_values=discrete_values, sigma=sigma, loss_factor=loss_factor)
        )

        self.max_sigma = sigma
        self.max_loss_factor = loss_factor

    def forward(self, x):
        identity1 = x
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.bn1_2(self.conv1_2(x))
        identity1 = self.shortcut1(identity1)
        x = F.relu(x + identity1)
        x = F.max_pool2d(x, 2)
        x = F.dropout2d(x, p=0.2, training=self.training)

        identity2 = x
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = self.bn2_2(self.conv2_2(x))
        identity2 = self.shortcut2(identity2)
        x = F.relu(x + identity2)
        x = F.max_pool2d(x, 2)
        x = F.dropout2d(x, p=0.2, training=self.training)

        x = self.classifier(x)
        return x


def get_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST("Dataset", train=True, transform=transform, download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataset = datasets.MNIST("Dataset", train=False, transform=transform, download=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_dataloader, test_dataloader


def cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

if __name__ == "__main__":
    SIGMA = 2.0
    LOSS_FACTOR = 2.0
    model = Gewu(sigma=SIGMA, loss_factor=LOSS_FACTOR).to(device)

    train_dataloader, test_dataloader = get_dataloaders()

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    TOTAL_EPOCHS = 150
    SIGMA_WARMUP_EPOCHS = 40
    MAX_RETRY = 5
    ACC_DROP_THRESHOLD = 0.02

    warmup_steps = 5
    scheduler = cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=TOTAL_EPOCHS)

    last_sigma = 0
    last_loss_factor = 0
    last_acc = 0
    retry_count = 0
    retry_num = 0
    real_epoch = 0

    while real_epoch < TOTAL_EPOCHS:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        if retry_count == 0:
            target_sigma = min(model.max_sigma * real_epoch / SIGMA_WARMUP_EPOCHS, model.max_sigma)
            target_loss_factor = min(model.max_loss_factor * real_epoch / SIGMA_WARMUP_EPOCHS, model.max_loss_factor)
            current_sigma = target_sigma
            current_loss_factor = target_loss_factor
        else:
            current_sigma = last_sigma
            current_loss_factor = last_loss_factor

        for module in model.modules():
            if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
                module.sigma = current_sigma
                module.loss_factor = current_loss_factor

        for batch_idx, (imgs, targets) in enumerate(train_dataloader):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, targets)

            l2_reg = 0.0001 * sum(torch.norm(p) for p in model.parameters())
            loss = loss + l2_reg

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {real_epoch}, Batch: {batch_idx}, Loss: {running_loss / (batch_idx + 1):.3f}, '
                      f'Acc: {100. * correct / total:.2f}%, Sigma: {current_sigma:.4f}, Loss_factor: {current_loss_factor:.4f}')

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, targets in test_dataloader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        current_acc = 100. * correct / total
        print(
            f"Epoch: {real_epoch}, 验证准确率: {current_acc:.2f}%, Current Sigma: {current_sigma:.4f}, Current Loss_factor: {current_loss_factor:.4f}")

        if retry_count == 0 and real_epoch > 0 and current_sigma > last_sigma and current_loss_factor > last_loss_factor:
            acc_drop = last_acc - current_acc
            if acc_drop > ACC_DROP_THRESHOLD * 100:
                if retry_num < MAX_RETRY:
                    print(
                        f"准确率下降{acc_drop:.2f}%超过阈值，回退到sigma={last_sigma:.4f},loss_factor={last_loss_factor:.4f},重新训练 (retry {retry_num + 1}/{MAX_RETRY})")
                    retry_count += 1
                    retry_num += 1
                    continue
                else:
                    print(f"达到最大重试次数{MAX_RETRY}，强制接受新的sigma和loss_factor值")
                    retry_num = 0
                    last_sigma = current_sigma
                    last_loss_factor = current_loss_factor
                    last_acc = current_acc
            else:
                print("准确率下降在可接受范围内,继续增大误差")
                last_sigma = current_sigma
                last_loss_factor = current_loss_factor
                last_acc = current_acc
                retry_num = 0
        elif retry_count > 0:
            print("重新训练新误差")
            retry_count = 0
            continue
        else:
            last_sigma = current_sigma
            last_loss_factor = current_loss_factor
            last_acc = current_acc

        scheduler.step()

        if current_sigma == SIGMA and current_acc >= 99.0:
            a0 = input(f"当前准确率为：{current_acc}%,是否保存当前模型？保存：y 不保存：n")
            if a0.lower() == "y":
                torch.save(model.state_dict(), f'model_weights_0429_test1.pth')
                print(f"已保存：model_weights_0429_test1.pth")
            else:
                pass

        real_epoch += 1

