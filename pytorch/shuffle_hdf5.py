import h5py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import sys

max_in_memory = 200000

with h5py.File(sys.argv[1], 'r') as h5_file:
    labels = h5_file['labels']
    images_shape = h5_file['images'].shape
    labels_shape = h5_file['labels'].shape
    n = len(labels)

# n = 300
idx = np.arange(n)
np.random.shuffle(idx)

subIdx = np.arange(0, n, max_in_memory)
subIdx = list(subIdx) + [n]
print('subIdx:', subIdx)

with h5py.File(sys.argv[2], 'w') as f:
    images_dset = f.create_dataset("images", images_shape, dtype='uint8', compression="gzip", compression_opts=4)
    labels_dset = f.create_dataset("labels", labels_shape, dtype='uint8', compression="gzip", compression_opts=4)

    for i in range(len(subIdx) - 1):
        print('Block %d/%d ...' % (i + 1, len(subIdx) - 1))
        images = np.empty((0, images_shape[1], images_shape[2], images_shape[3]), np.uint8)
        labels = np.empty((0, labels_shape[1]), np.uint8)
        idx2 = idx[subIdx[i] : subIdx[i+1]]

        for j in range(len(subIdx) - 1):
            print('  Loading images and labels from block %d/%d ...' % (j + 1, len(subIdx) - 1))
            with h5py.File(sys.argv[1], 'r') as h5_file:
                images_in = np.array(h5_file['images'][subIdx[j] : subIdx[j+1]])
                labels_in = np.array(h5_file['labels'][subIdx[j] : subIdx[j+1]])
            print('  Extracting images and labels at indices in this block ...')
            indices = idx2[np.where(np.logical_and(idx2 >= subIdx[j], idx2 < subIdx[j+1]))[0]]
            indices = indices - subIdx[j]
            images = np.vstack((images, images_in[indices]))
            labels = np.vstack((labels, labels_in[indices]))
            print('  images, labels:', images.shape, labels.shape)
        print('  Storing images and labels ...')
        images_dset[subIdx[i] : subIdx[i+1]] = images
        labels_dset[subIdx[i] : subIdx[i+1]] = labels
