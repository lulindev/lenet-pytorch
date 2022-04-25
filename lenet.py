import os
import random

import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm
import wandb


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1) log_softmax를 수행해야하지만 loss function에 포함되어 있으므로 생략.
        return x


class CustomLeNet(nn.Module):
    def __init__(self):
        super(CustomLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, bias=False)
        self.bn1 = nn.BatchNorm2d(10)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, bias=False)
        self.bn2 = nn.BatchNorm2d(20)
        self.maxpool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(20 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1) log_softmax를 수행해야하지만 loss function에 포함되어 있으므로 생략.
        return x


def train(model, trainloader, criterion, optimizer, device):
    model.train()

    train_loss = torch.zeros(1, device=device)
    correct = torch.zeros(1, dtype=torch.int64, device=device)
    for images, targets in trainloader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss
        pred = torch.argmax(outputs, dim=1)
        correct += torch.eq(pred, targets).sum()

    train_loss /= len(trainloader)
    accuracy = correct / len(trainloader.dataset) * 100
    return train_loss.item(), accuracy.item()


def evaluate(model, testloader, criterion, device):
    model.eval()

    test_loss = torch.zeros(1, device=device)
    correct = torch.zeros(1, dtype=torch.int64, device=device)
    for images, targets in testloader:
        images, targets = images.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(images)
        test_loss += criterion(outputs, targets)
        pred = torch.argmax(outputs, dim=1)
        correct += torch.eq(pred, targets).sum()

    test_loss /= len(testloader)
    accuracy = correct / len(testloader.dataset) * 100
    return test_loss.item(), accuracy.item()


if __name__ == '__main__':
    # 0. Hyper parameters
    hyper_parameters = {
        'batch_size': 256,
        'epoch': 10,
        'lr': 0.01,
        'reproducibility': True,
        'num_workers': 2,
        'pin_memory': True,
        'prefetch_factor': 30000,
        'persistent_workers': True,
    }
    wandb.init(project='test', entity='synml', config=hyper_parameters)
    config = wandb.config

    # Pytorch reproducibility
    if config.reproducibility:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        np.random.seed(0)
        random.seed(0)
        torch.use_deterministic_algorithms(True)

    # 1. Dataset, Dataloader
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='data', train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers,
    )

    # 2. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet().to(device)
    model_name = model.__str__().split('(')[0]

    # 3. Loss function, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RAdam(model.parameters(), config.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0, config.epoch)

    # 4. Collect gradients and parameters using wandb.
    wandb.watch(model, log='all', log_freq=10)

    # 5. Train and Test
    best_accuracy = 0
    for eph in tqdm.tqdm(range(config.epoch), desc='Epoch'):
        train_loss, train_accuracy = train(model, trainloader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, testloader, criterion, device)
        scheduler.step()

        # Log to wandb
        wandb.log({
            'Train_loss': train_loss,
            'Train_accuracy': train_accuracy,
            'Test_loss': test_loss,
            'Test_accuracy': test_accuracy,
            'lr': optimizer.param_groups[0]['lr'],
        })

        # Save model weight
        if test_accuracy > best_accuracy:
            wandb.summary['best_accuracy'] = test_accuracy
            os.makedirs('weights', exist_ok=True)
            torch.save(model.state_dict(), os.path.join('weights', f'{model_name}_best.pth'))
            best_accuracy = test_accuracy

    # Finish wandb
    wandb.finish()
