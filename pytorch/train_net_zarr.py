# Use tensors to speed up loading data onto the GPU during training.

import zarr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torchdata
import torch.multiprocessing
import sys

#torch.multiprocessing.set_start_method('spawn')

class ZarrDataset(torchdata.Dataset):
    def __init__(self, file_path, start_idx, end_idx):
        super(ZarrDataset, self).__init__()
        f = zarr.open(file_path, 'r')
        self.data = torch.from_numpy(np.array(f.get('images')[start_idx : end_idx]))
        print(self.data.dtype)
        self.target = torch.from_numpy(np.array(f.get('labels')[start_idx : end_idx])).to(torch.int32) #.astype('int32'))
        print("Loaded data")

    def __getitem__(self, index):
        image = self.data[index,:,:].to(torch.float32) * 1/256.0
        # Pytorch uses NCHW format
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
#        self.drop2d = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(50 * 12 * 12, 500)
#        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(500, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
#        x = self.pool(F.relu(self.drop2d(self.conv2(x))))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
#        x = self.drop1(x)
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

if len(sys.argv) < 3:
    print('ERROR: Not enough input arguments!')
    print('Usage: python train_net3.py pathToTrainingSet.h5 pathToTestSet.h5')
    exit(-1)

f = zarr.open(sys.argv[1], 'r')
num_train = len(f['images'])
num_epochs = 5
max_in_memory = 120000
repeats = 1
early_stop_loss = 0.0000001
start_idx = 0
end_idx = max_in_memory
iter_per_epoch = int(np.ceil(num_train / float(max_in_memory)))
indices = np.arange(0, num_train, max_in_memory)
indices = list(indices) + [num_train]
print('iter_per_epoch:', iter_per_epoch)
print(indices)

test_set = ZarrDataset(sys.argv[2], 0, 20000)
test_loader = torchdata.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=4)

# Use GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)
# Create the network.
net = Net(test_set.data.shape[-1])
print('Copying network to GPU ...')
net.to(device)
print(net)

# Define the loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
# optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)

# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

accuracy = eval(net, test_loader, device)
accuracies = []
accuracies.append(accuracy)

early_stop = False
losses = []
loss = None
print('Training ...')

for epoch in range(num_epochs):
    print('epoch: %d/%d' % (epoch + 1, num_epochs))
    net.train()

 #   scheduler.step()
    for param_group in optimizer.param_groups:
        print('learning rate:', param_group['lr'])

    for j in range(iter_per_epoch):
        print('Iteration: %d/%d' % (j + 1, iter_per_epoch))
        print('Loading data block [%d, %d] ...' % (indices[j], indices[j + 1]))
        dset = []
        train_loader = []
        dset = ZarrDataset(sys.argv[1], indices[j], indices[j + 1])
        train_loader = torchdata.DataLoader(dset, batch_size=64, shuffle=True, num_workers=4)
        running_loss = 0.0

        for r in range(repeats):
            if r > 1:
                print('Repeat: %d/%d' % (r + 1, repeats))
            for i, data in enumerate(train_loader):
                loss = train(net, criterion, optimizer, data, device)

                # print statistics
                running_loss += loss.item()
                if i % 200 == 199:
                    print('Epoch: %d, batch: %5d, loss: %.5f' %
                          (epoch + 1, i + 1, running_loss / 1000))
                    losses.append(running_loss)
                    if running_loss / 1000 < early_stop_loss:
                        print('Reached loss threshold for early stopping: %.5f', early_stop_loss)
                        early_stop = True
                        break
                    running_loss = 0.0
            # Evaluate the network on the test dataset.
            accuracy = eval(net, test_loader, device)
            accuracies.append(accuracy)
            model_path = 'model_' + str(accuracy) + '.pwf'
            torch.save(net.state_dict(), model_path)
            net.train()
            if early_stop:
                break
        if early_stop:
            break
    if early_stop:
        break

    # Evaluate the network on the test dataset.
    accuracy = eval(net, test_loader, device)
    accuracies.append(accuracy)
    model_path = 'model_' + str(accuracy) + '.pwf'
    torch.save(net.state_dict(), model_path)

print('Finished Training')

model_path = 'model.pwf'
torch.save(net.state_dict(), model_path)

with open('loss_stats.txt', 'w') as f:
    for l in losses:
        f.write("%s\n" % str(l))
with open('accuracy_stats.txt', 'w') as f:
    for a in accuracies:
        f.write("%s\n" % str(a))
