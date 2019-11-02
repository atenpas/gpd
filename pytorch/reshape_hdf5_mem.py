import h5py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import sys

h5_file = h5py.File(sys.argv[1], 'r')
images = h5_file['images']
labels = h5_file['labels']

num = int(sys.argv[3])

max_in_memory = 100000
indices = np.arange(0, num, max_in_memory)
indices = list(indices) + [num]
iters = int(np.ceil(num / float(max_in_memory)))
print('indices:', indices)
print('iters:', iters)
print(images.shape)
print(labels.shape)
print(images[0,0,0].shape)
#exit(-1)

with h5py.File(sys.argv[2]) as h5_file_out:
    print ('Creating HDF5 datasets ...')
    dset_images = h5_file_out.create_dataset('images', [num] + list(images.shape[1:]) + [images[0,0,0].shape[0]], compression="gzip", compression_opts=4)
    dset_labels = h5_file_out.create_dataset('labels', [num] + list(labels.shape[1:]), compression="gzip", compression_opts=4)

    for i in range(iters):
        print ('i: %d' % i)
        print ('Copying %d images ...' % (indices[i + 1] - indices[i]))
        dset_images[indices[i] : indices[i + 1], :, :] = images[indices[i] : indices[i + 1], :, :]
        print ('Copying %d labels ...' % (indices[i + 1] - indices[i]))
        dset_labels[indices[i] : indices[i + 1], :] = labels[indices[i] : indices[i + 1], :]

