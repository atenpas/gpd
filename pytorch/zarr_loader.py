import zarr
import torch
import sys
import torch.utils.data as torchdata
#import torch.multiprocessing
import numpy as np

#torch.multiprocessing.set_start_method('spawn')

class ZarrDataset(torchdata.Dataset):

    def __init__(self, file_path):
        super(ZarrDataset, self).__init__()
        root = zarr.open(file_path, mode='r')
        self.images = root['images']
        self.labels = root['labels']

    def __getitem__(self, index):
        return (torch.from_numpy(np.array(self.images[index])).float(),
                torch.from_numpy(np.array(self.labels[index])).int())

    def __len__(self):
        return self.images.shape[0]

train_dset = ZarrDataset(sys.argv[1])
train_loader = torchdata.DataLoader(train_dset, batch_size=64, shuffle=True, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

print('Iterating over dataset ...')

for i_batch, sample_batched in enumerate(train_loader):
    print(i_batch, sample_batched[0].size(), sample_batched[1].size())
    sample_batched[0] = sample_batched[0].to(device)
    sample_batched[1] = sample_batched[1].to(device)

    # observe 4th batch and stop.
    if i_batch == 100:
        break
