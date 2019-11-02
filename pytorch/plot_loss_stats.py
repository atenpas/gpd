import matplotlib.pyplot as plt
import numpy as np
import sys

with open(sys.argv[1], 'r') as f:
    losses = [line.rstrip() for line in f]
losses = np.array([float(x) for x in losses])

iter = np.arange(losses.shape[0])

fig, ax = plt.subplots()
ax.plot(iter, losses/100)
# plt.xticks(np.arange(0, 1, step=0.2))
plt.show()
