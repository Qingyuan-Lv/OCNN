import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import copy
import time
import os
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_recall_curve, average_precision_score
from thop import profile
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
            QuantizedLinear(64 * 7 * 7, 256, bias=True, discrete_values=discrete_values, sigma=sigma,
                            loss_factor=loss_factor),
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


class Compression():
    def __init__(self, model, model_path, sparsity_lambda=0.0001, pruned_percentage=0.1, distillation_miu=0.001,
                 distillation_temperature=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model_path = model_path
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.sparsity_lambda = sparsity_lambda
        self.true_model = copy.deepcopy(self.model).to(self.device)
        self.pruned_model = copy.deepcopy(self.true_model).to(self.device)
        self.masks = {}
        self.train_dataloader, self.test_dataloader = get_dataloaders()
        self.pruned_percentage = pruned_percentage
        self.distillation_miu = distillation_miu
        self.distillation_temperature = distillation_temperature
        self.metrics = {}
        self.pruning_iteration_count = 0

    def calculate_model_size(self, model):
        torch.save(model.state_dict(), "temp_model.pth")
        size_mb = os.path.getsize("temp_model.pth") / (1024 * 1024)
        os.remove("temp_model.pth")
        return size_mb

    def calculate_flops(self, model, input_shape=(1, 1, 28, 28)):
        input_data = torch.randn(input_shape).to(self.device)
        flops, params = profile(model, inputs=(input_data,), verbose=False)
        return flops, params

    def calculate_inference_time(self, model, input_shape=(1, 1, 28, 28), num_iterations=100):
        input_data = torch.randn(input_shape).to(self.device)
        with torch.no_grad():
            for _ in range(100):
                _ = model(input_data)

        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input_data)
        end_time = time.time()

        avg_time = (end_time - start_time) / num_iterations
        return avg_time

    def calculate_map(self, model, num_classes=10):
        model.eval()
        all_targets = []
        all_probas = []

        with torch.no_grad():
            for data in self.test_dataloader:
                imgs, targets = data
                imgs, targets = imgs.to(self.device), targets.to(self.device)
                outputs = model(imgs)
                probas = F.softmax(outputs, dim=1)

                all_targets.append(targets.cpu().numpy())
                all_probas.append(probas.cpu().numpy())

        all_targets = np.concatenate(all_targets, axis=0)
        all_probas = np.concatenate(all_probas, axis=0)

        target_one_hot = np.zeros((all_targets.size, num_classes))
        for i in range(all_targets.size):
            target_one_hot[i, all_targets[i]] = 1

        ap_list = []
        for class_idx in range(num_classes):
            ap = average_precision_score(target_one_hot[:, class_idx], all_probas[:, class_idx])
            ap_list.append(ap)

        mAP = np.mean(ap_list)
        return mAP, ap_list

    def calculate_compression_ratio(self, original_model, pruned_model):
        original_params = sum(p.numel() for p in original_model.parameters())
        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        compression_ratio = (1 - pruned_params / original_params) * 100
        return original_params, pruned_params, compression_ratio

    def evaluate_model_metrics(self, model, description="模型"):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_dataloader:
                imgs, targets = data
                imgs, targets = imgs.to(self.device), targets.to(self.device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = correct / total if total > 0 else 0

        flops, params = self.calculate_flops(model)

        model_size = self.calculate_model_size(model)

        inference_time = self.calculate_inference_time(model)

        mAP, ap_per_class = self.calculate_map(model)

        metrics = {
            "accuracy": accuracy * 100,
            "flops": flops,
            "model_size_mb": model_size,
            "inference_time_s": inference_time,
            "inference_time_10_4s": inference_time * 10 ** 4,
            "mAP": mAP * 100,
            "ap_per_class": ap_per_class
        }

        # 打印指标
        print(f"\n===== {description}指标评估 =====")
        print(f"准确率: {metrics['accuracy']:.2f}%")
        print(f"模型大小: {metrics['model_size_mb']:.2f}MB")
        print(f"FLOPs: {metrics['flops'] / 1e6:.2f}M")
        print(f"mAP: {metrics['mAP']:.2f}%")
        print(f"推理速度: {metrics['inference_time_s'] * 1000:.4f}ms ({metrics['inference_time_10_4s']:.2f}×10⁻⁴s)")
        return metrics

    def sparse_train_and_apply_pruning_and_knowledge_distillation(self, epochs_sparse_train=2, EPOCHS_sparse_train=4,
                                                                  epochs_knowledge_distillation=2,
                                                                  EPOCHS_knowledge_distillation=4, TOTAL_EPOCHS=1):
        print("评估原始模型性能...")
        original_metrics = self.evaluate_model_metrics(self.true_model, "原始模型")
        self.metrics['原始模型'] = original_metrics
        acc_true_model = original_metrics["accuracy"] / 100
        pruned_layer_names = ["bn2_2", "classifier.3"]
        current_pruned_percentage = self.pruned_percentage

        print("\n--- 记录指定层的初始权重数量 (在任何剪枝操作前) ---")
        initial_weights_label = "Initial"
        layers_to_track_weights_initially = ['classifier.2', 'conv2_2', 'classifier.6', 'shortcut2']
        for lname in layers_to_track_weights_initially:
            module_found = False
            for model_layer_name, mod in self.pruned_model.named_modules():
                if model_layer_name == lname:
                    if hasattr(mod, 'weight') and mod.weight is not None:
                        count = mod.weight.numel()
                        print(f"层 {lname} 的初始权重数量: {count}")
                    else:
                        print(f"层 {lname} (初始) 无权重数据或权重为None")
                    module_found = True
                    break
            if not module_found:
                print(f"层 {lname} (初始) 在模型中未找到")

        for layer in pruned_layer_names:
            pruned_epochs = 0
            while pruned_epochs <= TOTAL_EPOCHS:
                self.pruning_iteration_count += 1
                current_iteration_label = f"Lyr: {layer}, Iter: {pruned_epochs + 1}"

                print(
                    f"\n===== 剪枝层 {layer} 剪枝轮次 {pruned_epochs + 1}/{TOTAL_EPOCHS + 1}, 当前剪枝率: {current_pruned_percentage * 100:.1f}% =====")
                loss_func_sparse_train = nn.CrossEntropyLoss()
                optimizer_sparse_train = torch.optim.AdamW(self.pruned_model.parameters(), lr=0.001, weight_decay=0.01)
                warmup_steps = 2  # 预热epochs数
                scheduler_sparse_train = cosine_schedule_with_warmup(optimizer_sparse_train,
                                                                     num_warmup_steps=warmup_steps,
                                                                     num_training_steps=EPOCHS_sparse_train)
                all_gamma = []
                channel_maps = {}
                best_acc_sparse_train = 0
                real_epoch_sparse_train = 0
                MAX_RETRY = 5
                ACC_DROP_THRESHOLD = 0.01

                last_sigma = 0
                last_loss_factor = 0
                last_acc = 0
                retry_count = 0
                retry_num = 0

                gamma_module_to_record = None
                gamma_before_sparse_train = None
                for name, module in self.pruned_model.named_modules():
                    if name == layer and isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        gamma_module_to_record = module
                        break

                if gamma_module_to_record and layer == "bn2_2":
                    gamma_before_sparse_train = copy.deepcopy(gamma_module_to_record.weight.data.cpu().numpy())
                    print(
                        f"记录层 {layer} (迭代 {self.pruning_iteration_count}) 稀疏训练前的gamma值 (数量: {len(gamma_before_sparse_train)})")
                elif layer != "bn2_2":
                    print(f"层 {layer} (初始) 无权重数据或权重为None")

                _gamma_val_before = None
                _bn_name_for_plot = None
                if gamma_module_to_record:
                    if layer == "bn2_2":
                        _gamma_val_before = copy.deepcopy(gamma_module_to_record.weight.data.cpu().numpy())
                        _bn_name_for_plot = f"{layer} (Iter {self.pruning_iteration_count})"
                        print(f"记录层 {_bn_name_for_plot} 稀疏训练前的gamma值 (数量: {len(_gamma_val_before)})")
                if _gamma_val_before is not None:
                    print(f"记录层 {_bn_name_for_plot} 稀疏训练前的gamma值 (数量: {len(_gamma_val_before)})")

                while real_epoch_sparse_train <= EPOCHS_sparse_train:
                    if retry_count == 0:
                        current_sigma = min(self.pruned_model.max_sigma * real_epoch_sparse_train / epochs_sparse_train,
                                            self.pruned_model.max_sigma)
                        current_loss_factor = min(
                            self.pruned_model.max_loss_factor * real_epoch_sparse_train / epochs_sparse_train,
                            self.pruned_model.max_loss_factor)
                    else:
                        current_sigma = last_sigma
                        current_loss_factor = last_loss_factor

                    for module in self.pruned_model.modules():
                        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
                            module.sigma = current_sigma
                            module.loss_factor = current_loss_factor
                    self.pruned_model.train()
                    for data in self.train_dataloader:
                        gamma_L1 = 0
                        imgs, targets = data
                        imgs, targets = imgs.to(self.device), targets.to(self.device)
                        optimizer_sparse_train.zero_grad()
                        output = self.pruned_model(imgs)
                        loss = loss_func_sparse_train(output, targets)
                        for name, module in self.pruned_model.named_modules():
                            if name == layer:
                                gamma_L1 += torch.sum(torch.abs(module.weight.data))
                                l2_reg = 0.0001 * sum(torch.norm(p) for p in module.weight.data)
                        loss = loss + self.sparsity_lambda * gamma_L1 + l2_reg
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                        optimizer_sparse_train.step()
                    with torch.no_grad():
                        self.pruned_model.eval()
                        total = 0
                        correct = 0
                        for data in self.test_dataloader:
                            imgs, targets = data
                            imgs, targets = imgs.to(self.device), targets.to(self.device)
                            output = self.pruned_model(imgs)
                            _, predicted = torch.max(output, dim=1)
                            total += targets.size(0)
                            correct += (predicted == targets).sum().item()
                        current_acc = correct / total
                        print(
                            f"剪枝层 {layer},pruned_epochs{pruned_epochs},稀疏训练,epoch{real_epoch_sparse_train},sigma={current_sigma},loss_factor={current_loss_factor},稀疏训练准确率{100 * current_acc:.2f}%")
                    if retry_count == 0 and real_epoch_sparse_train > 0 and current_sigma > last_sigma and current_loss_factor > last_loss_factor:
                        acc_drop = last_acc - current_acc
                        if acc_drop > ACC_DROP_THRESHOLD:
                            if retry_num < MAX_RETRY:
                                print(
                                    f"准确率下降{100 * acc_drop:.2f}%超过阈值，回退到sigma={last_sigma:.4f},loss_factor={last_loss_factor:.4f},重新训练 (retry {retry_num + 1}/{MAX_RETRY})")
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
                    scheduler_sparse_train.step()
                    real_epoch_sparse_train += 1
                    if current_sigma == self.pruned_model.max_sigma and current_loss_factor == self.pruned_model.max_loss_factor and current_acc > best_acc_sparse_train:
                        best_acc_sparse_train = current_acc
                        best_stat_sparse_train = copy.deepcopy(self.pruned_model.state_dict())
                        print(f"保存当前最佳稀疏训练模型，准确率: {100 * best_acc_sparse_train:.2f}%（已达到目标误差）")
                self.pruned_model.load_state_dict(best_stat_sparse_train)
                print(f"已恢复最佳稀疏训练模型，准确率: {100 * best_acc_sparse_train:.2f}%")

                gamma_after_sparse_train = None
                if gamma_module_to_record and layer == "bn2_2":
                    refetched_gamma_module = None
                    for name, module_in_loop in self.pruned_model.named_modules():
                        if name == layer and isinstance(module_in_loop, (nn.BatchNorm2d, nn.BatchNorm1d)):
                            refetched_gamma_module = module_in_loop
                            break
                    if refetched_gamma_module:
                        gamma_after_sparse_train = copy.deepcopy(refetched_gamma_module.weight.data.cpu().numpy())
                        print(
                            f"记录层 {layer} (迭代 {self.pruning_iteration_count}) 稀疏训练后的gamma值 (数量: {len(gamma_after_sparse_train)})")
                if gamma_after_sparse_train is not None:
                    print(f"记录层 {layer} (迭代 {self.pruning_iteration_count}) 稀疏训练后的gamma值 (数量: {len(gamma_after_sparse_train)})")

                for name, module in self.pruned_model.named_modules():
                    if name == layer:
                        gamma = torch.abs(module.weight.data)
                        all_gamma.append(gamma.view(-1))
                all_gamma = torch.cat(all_gamma)
                total_channels = len(all_gamma)
                num_channels_to_prune = int(total_channels * current_pruned_percentage)
                sorted_gammas, _ = torch.sort(all_gamma)
                gamma_threshold = sorted_gammas[num_channels_to_prune].item()
                for name, module in self.pruned_model.named_modules():
                    if name == layer:
                        gamma = torch.abs(module.weight.data)
                        self.masks[name] = (gamma >= gamma_threshold)
                        channel_maps[name] = self.masks[name].nonzero().squeeze(1)
                        module.weight = nn.Parameter(module.weight[channel_maps[name]])
                        module.bias = nn.Parameter(module.bias[channel_maps[name]])
                        module.running_mean = module.running_mean[channel_maps[name]]
                        module.running_var = module.running_var[channel_maps[name]]
                        module.num_features = len(channel_maps[name])
                if layer == "bn2_2":
                    for name, module in self.pruned_model.named_modules():
                        if name == "conv2_2":
                            module.weight = nn.Parameter(module.weight[channel_maps["bn2_2"]])
                            module.out_channels = len(channel_maps["bn2_2"])
                            module.bias = nn.Parameter(module.bias[channel_maps["bn2_2"]])
                        elif name == "shortcut2":
                            module.weight = nn.Parameter(module.weight[channel_maps["bn2_2"]])
                            module.out_channels = len(channel_maps["bn2_2"])
                            module.bias = nn.Parameter(module.bias[channel_maps["bn2_2"]])
                        elif name == "classifier.2":
                            in_indices = channel_maps["bn2_2"]
                            H, W = 7, 7
                            flat_indices = []
                            for c in in_indices:
                                start_idx = c * H * W
                                for i in range(H * W):
                                    flat_indices.append(start_idx + i)
                            flat_indices = torch.tensor(flat_indices, device=module.weight.device)
                            new_weight = module.weight[:, flat_indices]
                            module.weight = nn.Parameter(new_weight)
                            module.in_features = len(flat_indices)
                elif layer == "classifier.3":
                    for name, module in self.pruned_model.named_modules():
                        if name == "classifier.2":

                            out_indices = channel_maps["classifier.3"]

                            new_weight = module.weight[out_indices]
                            module.weight = nn.Parameter(new_weight)
                            module.out_features = len(out_indices)
                            module.bias = nn.Parameter(module.bias[channel_maps["classifier.3"]])
                        elif name == "classifier.6":
                            module.weight = nn.Parameter(module.weight[:, channel_maps["classifier.3"]])
                            module.in_features = len(channel_maps["classifier.3"])
                else:
                    pass
                original_params, pruned_params, compression_ratio = self.calculate_compression_ratio(self.true_model,
                                                                                                     self.pruned_model)
                print(
                    f"剪枝层 {layer}, 当前剪枝率: {current_pruned_percentage * 100:.2f}%, 参数量: {original_params} -> {pruned_params}, 实际压缩率: {compression_ratio:.2f}%")

                print("\n--- 记录特定层权重数量变化 ---")
                layers_to_track_weights = ['classifier.2', 'conv2_2', 'classifier.6', 'shortcut2']
                for lname in layers_to_track_weights:
                    module_found = False
                    for model_layer_name, mod in self.pruned_model.named_modules():
                        if model_layer_name == lname:
                            if hasattr(mod, 'weight') and mod.weight is not None:
                                count = mod.weight.numel()
                                print(f"层 {lname} 剪枝后权重数量: {count}")
                            else:
                                print(f"层 {lname} 无权重数据或权重为None")
                            module_found = True
                            break
                    if not module_found:
                        print(f"层 {lname} 在剪枝模型中未找到")

                print("\n===== 剪枝后，知识蒸馏前模型评估 =====")
                metrics_after_pruning = self.evaluate_model_metrics(self.pruned_model,
                                                                    f"剪枝后模型 (层: {layer}, 迭代: {pruned_epochs + 1})")
                acc_after_pruning = metrics_after_pruning["accuracy"]
                print(f"剪枝后，蒸馏前准确率: {acc_after_pruning:.2f}%")

                loss_func_knowledge_distillation = nn.CrossEntropyLoss()
                optimizer_knowledge_distillation = torch.optim.AdamW(self.pruned_model.parameters(), lr=0.001,
                                                                     weight_decay=0.01)
                warmup_steps = 2
                scheduler_knowledge_distillation = cosine_schedule_with_warmup(optimizer_knowledge_distillation,
                                                                               num_warmup_steps=warmup_steps,
                                                                               num_training_steps=EPOCHS_knowledge_distillation)

                best_acc_knowledge_distillation = 0
                real_epoch_knowledge_distillation = 0
                MAX_RETRY = 5
                ACC_DROP_THRESHOLD = 0.03
                last_sigma = 0
                last_loss_factor = 0
                last_acc = 0
                retry_count = 0
                retry_num = 0
                while real_epoch_knowledge_distillation <= EPOCHS_knowledge_distillation:
                    self.pruned_model.train()
                    if retry_count == 0:
                        current_sigma = min(
                            self.pruned_model.max_sigma * real_epoch_knowledge_distillation / epochs_knowledge_distillation,
                            self.pruned_model.max_sigma)
                        current_loss_factor = min(
                            self.pruned_model.max_loss_factor * real_epoch_knowledge_distillation / epochs_knowledge_distillation,
                            self.pruned_model.max_loss_factor)
                    else:
                        current_sigma = last_sigma
                        current_loss_factor = last_loss_factor

                    for module in self.pruned_model.modules():
                        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
                            module.sigma = current_sigma
                            module.loss_factor = current_loss_factor
                    self.pruned_model.train()
                    for data in self.train_dataloader:
                        imgs, targets = data
                        imgs, targets = imgs.to(self.device), targets.to(self.device)
                        optimizer_knowledge_distillation.zero_grad()
                        output_pruned_model = self.pruned_model(imgs)
                        loss_student_targets = loss_func_knowledge_distillation(output_pruned_model, targets)
                        with torch.no_grad():
                            self.true_model.eval()
                            self.true_model.sigma = current_sigma
                            self.true_model.loss_factor = current_loss_factor
                            output_true_model = self.true_model(imgs)
                        soft_teacher = F.softmax(output_true_model / self.distillation_temperature, dim=1)
                        soft_student = F.log_softmax(output_pruned_model / self.distillation_temperature, dim=1)
                        loss_student_teacher = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (
                                self.distillation_temperature * self.distillation_temperature)
                        loss = self.distillation_miu * loss_student_targets + (
                                1 - self.distillation_miu) * loss_student_teacher
                        l2_reg = 0.0001 * sum(torch.norm(p) for p in model.parameters())
                        loss += l2_reg
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                        optimizer_knowledge_distillation.step()
                    scheduler_knowledge_distillation.step()
                    with torch.no_grad():
                        self.pruned_model.eval()
                        total = 0
                        correct = 0
                        for data in self.test_dataloader:
                            imgs, targets = data
                            imgs, targets = imgs.to(self.device), targets.to(self.device)
                            output = self.pruned_model(imgs)
                            _, predicted = torch.max(output, dim=1)
                            total += targets.size(0)
                            correct += (predicted == targets).sum().item()
                        current_acc = correct / total
                        print(
                            f"剪枝层 {layer},pruned_epochs{pruned_epochs},知识蒸馏,epoch{real_epoch_knowledge_distillation},sigma={current_sigma},loss_factor={current_loss_factor},知识蒸馏准确率{100 * current_acc:.2f}%")

                    if retry_count == 0 and real_epoch_knowledge_distillation > 0 and current_sigma > last_sigma and current_loss_factor > last_loss_factor:
                        acc_drop = last_acc - current_acc
                        if acc_drop > ACC_DROP_THRESHOLD:
                            if retry_num < MAX_RETRY:
                                print(
                                    f"准确率下降{100 * acc_drop:.2f}%超过阈值，回退到sigma={last_sigma:.4f},loss_factor={last_loss_factor:.4f},重新训练 (retry {retry_num + 1}/{MAX_RETRY})")
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
                    scheduler_knowledge_distillation.step()
                    real_epoch_knowledge_distillation += 1
                    if current_sigma == self.pruned_model.max_sigma and current_loss_factor == self.pruned_model.max_loss_factor and current_acc > best_acc_knowledge_distillation:
                        best_acc_knowledge_distillation = current_acc
                        best_stat_knowledge_distillation = copy.deepcopy(self.pruned_model.state_dict())
                        print(
                            f"保存当前最佳知识蒸馏训练模型，准确率: {100 * best_acc_knowledge_distillation:.2f}%（已达到目标误差）")
                self.pruned_model.load_state_dict(best_stat_knowledge_distillation)
                print(f"已恢复最佳知识蒸馏训练模型，准确率: {100 * best_acc_knowledge_distillation:.2f}%")
                print("\n===== 知识蒸馏后模型评估 =====")
                distilled_metrics = self.evaluate_model_metrics(self.pruned_model,
                                                                f"蒸馏后模型(剪枝率:{current_pruned_percentage * 100:.1f}%)")
                self.metrics[f'distilled_{layer}_ep{pruned_epochs}'] = distilled_metrics

                acc_after_distillation = distilled_metrics["accuracy"]
                print(f"知识蒸馏后准确率记录: {acc_after_distillation:.2f}%")

                print(f"\n===== 与原始模型比较 =====")
                print(f"FLOPs减少: {(1 - distilled_metrics['flops'] / original_metrics['flops']) * 100:.2f}%")
                print(
                    f"模型大小减少: {(1 - distilled_metrics['model_size_mb'] / original_metrics['model_size_mb']) * 100:.2f}%")
                print(
                    f"推理速度提升: {(original_metrics['inference_time_s'] / distilled_metrics['inference_time_s']):.2f}倍")
                print(f"准确率变化: {distilled_metrics['accuracy'] - original_metrics['accuracy']:.2f}%")
                print(f"mAP变化: {distilled_metrics['mAP'] - original_metrics['mAP']:.2f}%")
                acc_decline = acc_true_model - best_acc_knowledge_distillation
                print(f"准确率下降: {acc_decline * 100:.2f}%")
                pruned_epochs += 1
        print("\n===== 最终模型评估 =====")
        final_metrics = self.evaluate_model_metrics(self.pruned_model, "最终模型")
        self.metrics['final'] = final_metrics

        print(f"\n===== 最终模型与原始模型比较 =====")
        print(f"FLOPs减少: {(1 - final_metrics['flops'] / original_metrics['flops']) * 100:.2f}%")
        print(f"模型大小减少: {(1 - final_metrics['model_size_mb'] / original_metrics['model_size_mb']) * 100:.2f}%")
        print(f"推理速度提升: {(original_metrics['inference_time_s'] / final_metrics['inference_time_s']):.2f}倍")
        print(f"准确率变化: {final_metrics['accuracy'] - original_metrics['accuracy']:.2f}%")
        print(f"mAP变化: {final_metrics['mAP'] - original_metrics['mAP']:.2f}%")

        print(f"经过迭代剪枝蒸馏，得到最终模型,准确率{final_metrics['accuracy']:.2f}%, "
              f"FLOPs: {final_metrics['flops'] / 1e6:.2f}M, "
              f"模型大小: {final_metrics['model_size_mb']:.2f}MB, "
              f"推理速度: {final_metrics['inference_time_10_4s']:.2f}×10⁻⁴s, "
              f"mAP: {final_metrics['mAP']:.2f}%")
        original_params, pruned_params, final_compression_ratio = self.calculate_compression_ratio(self.true_model,
                                                                                                   self.pruned_model)
        print(f"原始参数量: {original_params}")
        print(f"剪枝后参数量: {pruned_params}")
        print(f"实际压缩率: {final_compression_ratio:.2f}%")

        return self.pruned_model


if __name__ == "__main__":
    model = Gewu(sigma=2.0, loss_factor=2.0)
    model_path = "model_weights_0429_test1.pth"
    compression = Compression(model=model, model_path=model_path, sparsity_lambda=0.1, pruned_percentage=0.4,
                              distillation_miu=0.001, distillation_temperature=5)
    final_model = compression.sparse_train_and_apply_pruning_and_knowledge_distillation()
    torch.save(final_model, "compressed_model_test_0507.pth")

