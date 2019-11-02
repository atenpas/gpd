import h5py
import numpy as np
import sys
import zarr

print('Loading database ...')
h5_file = h5py.File(sys.argv[1])
images = h5_file.get('images')
labels = h5_file.get('labels')

root = zarr.open(sys.argv[2], mode='w')
print('Writing images of shape', images.shape, '...')
images_out = root.create_dataset('images', data=images, dtype=images.dtype) #, chunks=(5000,) + images.shape[1:])
print('Writing labels of shape', labels.shape, '...')
labels_out = root.create_dataset('labels', data=labels, dtype=labels.dtype)
