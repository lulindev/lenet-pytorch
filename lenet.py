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
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # log_softmax를 수행해야하지만 loss function에 포함됨.
        return x


def train(model, trainloader, creterion, optimizer, writer, epoch, device):
    model.train()
    for batch_idx, (images, targets) in enumerate(tqdm.tqdm(trainloader, desc='Batch', leave=False)):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = creterion(outputs, targets)
        loss.backward()
        optimizer.step()

        writer.add_scalar('Train loss', loss.item(), len(trainloader) * epoch + batch_idx)


def evaluate(model, testloader, creterion, device):
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, targets in tqdm.tqdm(testloader, desc='Eval', leave=False):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)

            test_loss += creterion(outputs, targets).item()

            pred = torch.argmax(F.log_softmax(outputs, dim=1), dim=1)
            correct += torch.eq(pred, targets).sum().item()

    test_loss /= len(testloader.dataset)
    test_accuracy = 100 * correct / len(testloader.dataset)
    return test_loss, test_accuracy


if __name__ == '__main__':
    # 0. Hyper parameters
    batch_size = 512
    epoch = 50
    lr = 0.0001

    # 1. Dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.MNIST(root='dataset', train=True, download=True,
                                          transform=transform)
    testset = torchvision.datasets.MNIST(root='dataset', train=False, download=True,
                                         transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size, num_workers=4)

    # 2. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet().to(device)

    # 3. Loss function, optimizer
    creterion = nn.CrossEntropyLoss()
    creterion_test = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. Tensorboard
    writer = torch.utils.tensorboard.SummaryWriter('runs')
    writer.add_graph(model, trainloader.__iter__().__next__()[0].to(device))
    writer.add_hparams({'Batch size': batch_size, 'Epoch': epoch, 'lr': lr}, {' ': 0})

    # 5. Train and test
    for eph in tqdm.tqdm(range(epoch), desc='Epoch'):
        train(model, trainloader, creterion, optimizer, writer, eph, device)

        test_loss, test_accuracy = evaluate(model, testloader, creterion_test, device)
        writer.add_scalars('Test', {'Test loss': test_loss, 'Test accuracy': test_accuracy}, eph)

    # 6. Save model
    torch.save(model.state_dict(), 'weights/lenet.pth')
