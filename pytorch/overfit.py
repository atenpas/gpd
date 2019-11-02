# Use tensors to speed up loading data onto the GPU during training.

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torchdata
import sys

class H5Dataset(torchdata.Dataset):
    def __init__(self, file_path, start_idx, end_idx):
        super(H5Dataset, self).__init__()
        with h5py.File(file_path, 'r') as h5_file:
            self.data = torch.from_numpy(np.array(h5_file.get('images')[start_idx : end_idx]).astype('float32'))
            self.target = torch.from_numpy(np.array(h5_file.get('labels')[start_idx : end_idx]).astype('int32'))
        print("Loaded data")

    def __getitem__(self, index):
        image = self.data[index,:,:]
        # ptorch uses NCHW format
        image = image.reshape((image.shape[2], image.shape[0], image.shape[1]))
        target = self.target[index,:][0]
        return (image, target)

    def __len__(self):
        return self.data.shape[0]

class Net(nn.Module):
    def __init__(self, input_channels):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 40, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(40, 80, 5)
        self.fc1 = nn.Linear(80 * 12 * 12, 1000)
        self.fc2 = nn.Linear(1000, 2)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 80 * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

with h5py.File(sys.argv[1], 'r') as db:
    num_train = len(db['images'])
print('Have', num_train, 'total training examples')
num_epochs = 60
max_in_memory = 10000

# Use GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load the training data.
print('Loading data ...')

# Create the network.
input_channels = int(sys.argv[3])
net = Net(input_channels)
print(net)

print('Copying network to GPU ...')
net.to(device)

# Define the loss function and optimizer.
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01)
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.05)
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.1)
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.01)
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.05)
# optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9, weight_decay=0.05)
# optimizer = optim.SGD(net.parameters(), lr=0.000001, momentum=0.9, weight_decay=0.01)

# optimizer = optim.Adam(net.parameters(), weight_decay=0.05)
# optimizer = optim.Adam(net.parameters(), weight_decay=0.01)
optimizer = optim.Adam(net.parameters(), weight_decay=0.005)

#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

dset = H5Dataset(sys.argv[1], 0, max_in_memory)
#train_loader = torchdata.DataLoader(dset, batch_size=64, shuffle=True, num_workers=2)
train_loader = torchdata.DataLoader(dset, batch_size=64, shuffle=True)

print('Training ...')

for epoch in range(num_epochs):
    print('epoch: %d/%d' % (epoch, num_epochs))
 #   scheduler.step()
    for param_group in optimizer.param_groups:
        print('learning rate:', param_group['lr'])

    running_loss = 0.0

    for i, data in enumerate(train_loader):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99: # print every 10 mini-batches
            print('epoch: %d, batch: %5d, loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 100))

print('Finished Training')

# Test the network on the test data.
test_set = H5Dataset(sys.argv[2], 0, 20000)
test_loader = torchdata.DataLoader(test_set, batch_size=64, shuffle=True)
correct = 0
total = 0
print('Testing the network on the test data ...')
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()

print('Accuracy of the network on the 20000 test images: %d %%' % (
    100 * correct / total))
