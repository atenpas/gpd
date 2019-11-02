import torch.multiprocessing as mp

# mp.set_start_method('spawn')

def train(model):
    pass

model = 1

num_processes = 4
processes = []
for rank in range(num_processes):
    p = mp.Process(target=train, args=(model,))
    p.start()
    processes.append(p)
for p in processes:
    p.join()
