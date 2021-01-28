import torch
import torch.utils.data
import torch.utils.tensorboard
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # log_softmax를 수행해야하지만 loss function에 포함되어 있으므로 생략.
        return x


class CustomLeNet(nn.Module):
    def __init__(self):
        super(CustomLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(20 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # log_softmax를 수행해야하지만 loss function에 포함되어 있으므로 생략.
        return x


def train(model, trainloader, criterion, optimizer, writer, epoch, device):
    model.train()
    for batch_idx, (images, targets) in enumerate(tqdm.tqdm(trainloader, desc='Train', leave=False)):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        writer.add_scalar('Train Loss', loss.item(), len(trainloader) * epoch + batch_idx)


def evaluate(model, testloader, criterion, device):
    model.eval()

    val_loss = 0
    correct = 0
    with torch.no_grad():
        for images, targets in tqdm.tqdm(testloader, desc='Eval', leave=False):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)

            val_loss += criterion(outputs, targets).item()

            pred = torch.argmax(F.log_softmax(outputs, dim=1), dim=1)
            correct += torch.eq(pred, targets).sum().item()

    val_loss /= len(testloader.dataset)
    accuracy = 100 * correct / len(testloader.dataset)
    return val_loss, accuracy


if __name__ == '__main__':
    # 0. Hyper parameters
    batch_size = 256
    epoch = 100
    lr = 0.001

    # 1. Dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.MNIST(root='dataset', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='dataset', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size, num_workers=4)

    # 2. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet().to(device)

    # 3. Loss function, optimizer
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. Tensorboard
    writer = torch.utils.tensorboard.SummaryWriter('runs/' + model.__str__().split('(')[0])
    writer.add_graph(model, trainloader.__iter__().__next__()[0].to(device))

    # 5. Train and test
    prev_accuracy = 0
    for eph in tqdm.tqdm(range(epoch), desc='Epoch'):
        train(model, trainloader, criterion, optimizer, writer, eph, device)

        val_loss, accuracy = evaluate(model, testloader, criterion, device)
        writer.add_scalar('Test Loss', val_loss, eph)
        writer.add_scalar('Test Accuracy', accuracy, eph)

        if accuracy > prev_accuracy:
            torch.save(model.state_dict(), '{}_best.pth'.format(model.__str__().split('(')[0]))
            prev_accuracy = accuracy
    writer.close()
