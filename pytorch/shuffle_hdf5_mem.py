import h5py
import h5py_cache as h5c
import numpy as np
import os
import sys

CHUNK_CACHE = 1024**2*4000

if len(sys.argv) < 4:
    print('Error: not enough input arguments!')
    print('Usage: python3 shuffle_h5_mem_2pass.py IN_H5 OUT_H5 STEPS')
    exit(-1)

fin = sys.argv[1]
fout = sys.argv[2]
steps = int(sys.argv[3])

with h5c.File(fin, 'r', chunk_cache_mem_size=CHUNK_CACHE) as f:
    images_shape = f['images'].shape
    labels_shape = f['labels'].shape
    n = labels_shape[0]
    max_in_memory = int(np.floor(n/steps))
    blocks = list(np.arange(0, n, max_in_memory)) + [labels_shape[0]]
    blocks = blocks[:-1]
    blocks[-1] = n
    m = len(blocks) - 1
    print('n:', n, 'm:', m, 'blocks:', blocks)
    chunk_shape = (1000,) + images_shape[1:]
    db_idx = np.random.permutation(np.repeat(np.arange(m), max_in_memory))
    print('db_idx:', len(db_idx))
    offsets = [0]*m

    # 1st pass: Assign instances to randomly chosen bins.
    print('1st pass: assign instances to randomly chosen bins ...')
    with h5c.File('temp.h5', 'w', chunk_cache_mem_size=CHUNK_CACHE) as db:
        for i in range(m):
            db.create_dataset('images' + str(i), (max_in_memory,) + images_shape[1:], dtype='uint8', chunks=chunk_shape)
            db.create_dataset('labels' + str(i), (max_in_memory,) + labels_shape[1:], dtype='uint8')

        for i in range(m):
            print('%d/%d: %d to %d' % (i+1, m, blocks[i], blocks[i+1]))
            images_in = f['images'][blocks[i] : blocks[i+1]]
            labels_in = f['labels'][blocks[i] : blocks[i+1]]
            idx = db_idx[blocks[i] : blocks[i+1]]
            print('  %d indices in this block' % len(idx))
            for j in range(m):
                in_bin = np.where(idx == j)[0]
                start = offsets[j]
                end = offsets[j] + len(in_bin)
                offsets[j] = end
                print('    %d indices in bin %d' % (len(in_bin), j))
                print('    start: %d, end: %d' % (start, end))
                db['images' + str(j)][start:end] = np.array(images_in)[in_bin]
                db['labels' + str(j)][start:end] = np.array(labels_in)[in_bin]

    # 2nd pass: Shuffle each bin.
    print('--------------------------------------------------------')
    print('2nd pass: shuffle each bin ...')
    with h5c.File(fout, 'w', chunk_cache_mem_size=CHUNK_CACHE) as fout:
        images_out = fout.create_dataset('images', images_shape, dtype='uint8', chunks=chunk_shape, compression='lzf')
        labels_out = fout.create_dataset('labels', labels_shape, dtype='uint8')

        with h5c.File('temp.h5', 'r', chunk_cache_mem_size=CHUNK_CACHE) as fin:
            for i in range(m):
                images_in = np.array(fin['images' + str(i)])
                labels_in = np.array(fin['labels' + str(i)])
                p = np.random.permutation(len(labels_in))
                k = len(p)
                print('%d/%d: %d to %d. k: %d' % (i+1, m, blocks[i], blocks[i+1], k))
                images_out[blocks[i] : blocks[i] + k] = images_in[p]
                labels_out[blocks[i] : blocks[i] + k] = labels_in[p]

