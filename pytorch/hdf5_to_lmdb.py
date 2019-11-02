import h5py
import lmdb
import numpy as np
import sys

h5_file = h5py.File(sys.argv[1])
data = h5_file.get('images')
target = h5_file.get('labels')

num = int(sys.argv[3])
data = data[:num]
target = target[:num]

map_size = data.nbytes * 10
env = lmdb.open(sys.argv[2], map_size=map_size)

for i in range(data.shape[0]):
    with env.begin(write=True) as txn:
        txn.put('X_' + str(i), data[i])
        txn.put('y_' + str(i), target[i])
    if i % 1000 == 0:
        print i, data.shape[0]
