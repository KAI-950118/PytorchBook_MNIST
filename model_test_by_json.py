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
import pyimgur

data_transforms = transforms.Compose([transforms.ToTensor()])
CLIENT_ID = "3fa55bc53a3cf95"

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
idx = 5432
test_data = test_dataset[idx][0]
print(test_data)
print(type(test_data))
print(test_dataset[idx][1])

title = str(idx) +"-"+ str(test_dataset[idx][1])

## upload the MNIST data to Imgur
# im = pyimgur.Imgur(CLIENT_ID)
# uploaded_image = im.upload_image(test_data, title=title)
# print(uploaded_image.title)
# print(uploaded_image.link)
# print(uploaded_image.type)

## method = "POST"
req = requests.post('http://127.0.0.1:5000/predict', json={'image': 'https://i.imgur.com/Y28rZho.png'})
print(json.loads(req.content))

## method = "GET"
# req = requests.get('http://127.0.0.1:5000/performance')
# print(json.loads(req.content)["performance"])