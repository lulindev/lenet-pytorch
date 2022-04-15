import os
import random

import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data
import torch.utils.tensorboard
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
    for images, targets in tqdm.tqdm(trainloader, desc='Train', leave=False):
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
    for images, targets in tqdm.tqdm(testloader, desc='Evaluate', leave=False):
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
    config = {
        'batch_size': 256,
        'epoch': 10,
        'lr': 0.01,
        'reproducibility': True,
        'num_workers': 2,
        'pin_memory': True,
        'prefetch_factor': 30000,
        'persistent_workers': True,
    }

    # Pytorch reproducibility
    if config['reproducibility']:
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
        config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        prefetch_factor=config['prefetch_factor'],
        persistent_workers=config['persistent_workers'],
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        prefetch_factor=config['prefetch_factor'],
        persistent_workers=config['persistent_workers'],
    )

    # 2. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet().to(device)
    model_name = model.__str__().split('(')[0]

    # 3. Loss function, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RAdam(model.parameters(), config['lr'])
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0, config['epoch'])

    # 4. Tensorboard
    writer = torch.utils.tensorboard.SummaryWriter(os.path.join('runs', model_name))
    writer.add_graph(model, trainloader.__iter__().__next__()[0].to(device))
    wandb.init(project='test', entity='synml', name=model_name, config=config)
    wandb.watch(model, criterion, log='all', log_freq=10)

    # 5. Train and Test
    best_accuracy = 0
    for eph in tqdm.tqdm(range(config['epoch']), desc='Epoch'):
        train_loss, train_accuracy = train(model, trainloader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, testloader, criterion, device)
        scheduler.step()

        # Write data to Tensorboard
        writer.add_scalar('Train/loss', train_loss, eph)
        writer.add_scalar('Train/accuracy', train_accuracy, eph)
        writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], eph)
        writer.add_scalar('Test/loss', test_loss, eph)
        writer.add_scalar('Test/accuracy', test_accuracy, eph)
        wandb.log({
            'Train': {'loss': train_loss, 'accuracy': train_accuracy, 'lr': optimizer.param_groups[0]['lr']},
            'Test': {'loss': test_loss, 'accuracy': test_accuracy},
        })

        # Save model weight
        if test_accuracy > best_accuracy:
            wandb.summary['best_accuracy'] = test_accuracy
            os.makedirs('weights', exist_ok=True)
            torch.save(model.state_dict(), os.path.join('weights', f'{model_name}_best.pth'))
            best_accuracy = test_accuracy

    # Close Tensorboard
    writer.close()
    wandb.finish()
