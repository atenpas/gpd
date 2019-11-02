import h5py
import torch
import sys
import torch.utils.data as torchdata
import torch.multiprocessing

torch.multiprocessing.set_start_method('spawn')

class H5Dataset(torchdata.Dataset):

    def __init__(self, file_path):
        super(H5Dataset, self).__init__()
        h5_file = h5py.File(file_path)
        self.data = h5_file.get('images')
        self.target = h5_file.get('labels')

    def __getitem__(self, index):
        return (torch.from_numpy(self.data[index]).float(),
                torch.from_numpy(self.target[index]).int())

    def __len__(self):
        return self.data.shape[0]

train_dset = H5Dataset(sys.argv[1])
train_loader = torchdata.DataLoader(train_dset, batch_size=64, shuffle=True, num_workers=0)

print('Iterating over dataset ...')

for i_batch, sample_batched in enumerate(train_loader):
    print(i_batch, sample_batched[0].size(), sample_batched[0].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        break
