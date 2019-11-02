import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from collections import OrderedDict
from network import Net, NetCCFFF

if len(sys.argv) < 4:
    print('ERROR: Not enough input arguments!')
    print('Usage: python torch_to_onxx.py pathToPytorchModel.pwf pathToONNXModel.onnx num_channels')
    exit(-1)

state_dict = torch.load(sys.argv[1])
#new_state_dict = OrderedDict()
#for k, v in state_dict.items():
#    name = k[7:]
#    new_state_dict[name] = v

input_channels = int(sys.argv[3])
net = Net(input_channels)
net.load_state_dict(state_dict)
print(net)

dummy_input = torch.randn(1, input_channels, 60, 60)
torch.onnx.export(net, dummy_input, sys.argv[2], verbose=True)
