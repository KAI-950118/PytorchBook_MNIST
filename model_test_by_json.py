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
from PIL import Image

data_transforms = transforms.Compose([transforms.ToTensor()])

## MNIST sample images "from https://gist.github.com/peterroelants/3a490905f5b022fea66e0553af51abb8"
# IMAGE_URLS = [
#     'https://i.imgur.com/SdYYBDt.png',  # 0
#     'https://i.imgur.com/Wy7mad6.png',  # 1
#     'https://i.imgur.com/nhBZndj.png',  # 2
#     'https://i.imgur.com/V6XeoWZ.png',  # 3
#     'https://i.imgur.com/EdxBM1B.png',  # 4
#     'https://i.imgur.com/zWSDIuV.png',  # 5
#     'https://i.imgur.com/Y28rZho.png',  # 6
#     'https://i.imgur.com/6qsCz2W.png',  # 7
#     'https://i.imgur.com/BVorzCP.png',  # 8
#     'https://i.imgur.com/vt5Edjb.png',  # 9
# ]

## loading test data
test_dataset = dsets.MNIST(root='./data', train=False, download=True)
idx = 7811
test_image = test_dataset[idx][0]
# print(type(test_image))
test_tensor = data_transforms(test_image).numpy()
# print(type(test_tensor))
test_list = test_tensor.tolist()
# print(type(test_list))

## Show the image
plt.imshow(test_tensor[0, ...])
# print(test_dataset[idx][1])
plt.show()


## method = "POST"  URL
# req = requests.post('http://127.0.0.1:5000/predict', json={'image': 'https://i.imgur.com/Y28rZho.png'})
# print(json.loads(req.content))
# dict = json.loads(req.content)
# print("Prediction:", dict["predictions"])
# print("Answer:", test_dataset[idx][1])

## method = "POST"  list
req = requests.post('http://127.0.0.1:5000/predict', json={'image': test_list})
# print(json.loads(req.content))
dict = json.loads(req.content)
print("Prediction:", dict["predictions"])
print("Answer:", test_dataset[idx][1])

## method = "GET"
# req = requests.get('http://127.0.0.1:5000/performance')
# print(json.loads(req.content)["performance"])
