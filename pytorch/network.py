import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNELS = [20, 50, 500]
#CHANNELS = [40, 100, 500]
#CHANNELS = [10, 20, 100]

#CHANNELS = [6, 16, 120, 84]
#CHANNELS = [12, 32, 120, 84]
#CHANNELS = [32, 32, 120, 84]

class NetCCFFF(nn.Module):
    def __init__(self, input_channels):
        super(NetCCFFF, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, CHANNELS[0], 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(CHANNELS[0], CHANNELS[1], 5)
        self.fc1 = nn.Linear(CHANNELS[1] * 12 * 12, CHANNELS[2])
        self.fc2 = nn.Linear(CHANNELS[2], CHANNELS[3])
        self.fc3 = nn.Linear(CHANNELS[3], 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net(nn.Module):
    def __init__(self, input_channels):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, CHANNELS[0], 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(CHANNELS[0], CHANNELS[1], 5)
        self.fc1 = nn.Linear(CHANNELS[1] * 12 * 12, CHANNELS[2])
        self.fc2 = nn.Linear(CHANNELS[2], 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, criterion, optimizer, data, device):
    # Get the inputs and transfer them to the CPU/GPU.
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Reset the parameter gradients.
    optimizer.zero_grad()

    # Forward + backward + optimize.
    outputs = model(inputs)
    loss = criterion(outputs, labels.long())
    loss.backward()
    optimizer.step()

    return loss

def eval(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    print('Testing the network on the test data ...')

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()

    accuracy = 100.0 * float(correct) / float(total)
    print('Accuracy of the network on the test set: %.3f%%' % (
        accuracy))

    return accuracy
