import h5py
import h5py_cache as h5c
import torch
import torch.utils.data as torchdata

CHUNK_CACHE_MEM_SIZE = 1024**2*4000

class H5Dataset(torchdata.Dataset):
    def __init__(self, file_path, start=0, end=None):
        super(H5Dataset, self).__init__()
        with h5c.File(file_path, 'r', chunk_cache_mem_size=CHUNK_CACHE_MEM_SIZE) as f:
            self.images = torch.from_numpy(f['images'][start : end])
            self.labels = torch.from_numpy(f['labels'][start : end]).to(torch.int32)
        print("Loaded images of shape {}, type {}, and labels of shape {}, type {}.".format(self.images.shape, self.images.dtype, self.labels.shape, self.labels.dtype))

    def __getitem__(self, index):
        image = self.images[index,:,:].to(torch.float32) * 1/256.0
        # Pytorch uses NCHW format
        image = image.permute(2, 0, 1)
        label = self.labels[index,:][0]
        return (image, label)

    def __len__(self):
        return self.labels.shape[0]
