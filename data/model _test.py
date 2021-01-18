import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import io
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import requests
import json

test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
idx = 6999
muteimg = test_dataset[idx][0].numpy()
plt.imshow(muteimg[0, ...])
print(test_dataset[idx][1])
plt.show()
# print(muteimg)

test_data = Variable(test_dataset[idx][0].unsqueeze(0))
#
req = requests.post('http://127.0.0.1:5000/predict') #, json={'image': test_data})
print(json.loads(req.content))

# req = requests.get('http://127.0.0.1:5000/performance')
# print(json.loads(req.content)["performance"])