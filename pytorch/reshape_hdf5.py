import h5py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import sys

class H5Dataset(data.Dataset):
    def __init__(self, file_path):
        super(H5Dataset, self).__init__()
        # with h5py.File(file_path, 'r') as h5_file:
        h5_file = h5py.File(file_path, 'r')
        self.data = h5_file.get('images')
        self.target = h5_file.get('labels')

    def __getitem__(self, index):
        return (torch.from_numpy(self.data[index,:,:,:]).float(),
                torch.from_numpy(self.target[index,:,:,:]).float())

    def __len__(self):
        return self.data.shape[0]

dset = H5Dataset(sys.argv[1])
num = int(sys.argv[3])

with h5py.File(sys.argv[2]) as h5_file_out:
    print ('Storing images ...')
    dset_small = h5_file_out.create_dataset('images', data=dset.data[:num], compression="gzip", compression_opts=4)
    print ('Storing labels ...')
    dset_small = h5_file_out.create_dataset('labels', data=dset.target[:num], compression="gzip", compression_opts=4)
    print(len(dset))
    print(len(dset_small))
