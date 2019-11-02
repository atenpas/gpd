# Use tensors to speed up loading data onto the GPU during training.

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torchdata
import torch.multiprocessing
import sys

#torch.multiprocessing.set_start_method('spawn')

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
        self.conv1 = nn.Conv2d(input_channels, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50 * 12 * 12, 500)
        self.fc2 = nn.Linear(500, 2)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 50 * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

with h5py.File(sys.argv[1], 'r') as db:
    num_train = len(db['images'])
print('Have', num_train, 'total training examples')
num_epochs = 10
max_in_memory = 50000
repeats = 2
early_stop_loss = 0.15
start_idx = 0
end_idx = max_in_memory
iter_per_epoch = int(np.ceil(num_train / float(max_in_memory)))
indices = np.arange(0, num_train, max_in_memory)
indices = list(indices) + [num_train]
print('iter_per_epoch:', iter_per_epoch)
print(indices)

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
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.01)
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.05)
# optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9, weight_decay=0.05)
# optimizer = optim.SGD(net.parameters(), lr=0.000001, momentum=0.9, weight_decay=0.01)

early_stop = False
print('Training ...')

for epoch in range(num_epochs):
    print('epoch: %d/%d' % (epoch, num_epochs))

    for j in range(iter_per_epoch):
        print('iter: %d/%d' % (j, iter_per_epoch))
        dset = H5Dataset(sys.argv[1], indices[j], indices[j + 1])
        #train_loader = torchdata.DataLoader(dset, batch_size=64, shuffle=True, num_workers=2)
        train_loader = torchdata.DataLoader(dset, batch_size=64, shuffle=True)

        running_loss = 0.0

        for r in range(repeats):
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
                    if running_loss / 100 < early_stop_loss:
                        print('reached loss threshold for early stopping: %.5f', early_stop_loss)
                        early_stop = True
                        break
                    running_loss = 0.0
            if early_stop:
                break
        if early_stop:
            break
    if early_stop:
        break

print('Finished Training')

model_path = raw_input("Enter the filename/path for the trained model: ")
torch.save(net.state_dict(), model_path)

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
