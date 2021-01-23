import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
# print(cuda.gpus)



image_size = 28
num_classes = 10
num_epochs = 2
batch_size = 64

## Data pre-process
train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
indices = range(len(test_dataset))
indices_val = indices[:5000]
indices_test = indices[5000:]

sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)

validation_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, sampler=sampler_val)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, sampler=sampler_test)

## image check
# idx = 150
# muteimg = train_dataset[idx][0].numpy()
# plt.imshow(muteimg[0, ...])
# print(train_dataset[idx][1])
# plt.show()

## CNN network
depth = [4, 8]
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, padding=2)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(depth[0], depth[1], 5, padding=2)
        self.fc1 = nn.Linear(image_size // 4 * image_size // 4 * depth[1], 512)
        self.fc2 = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, image_size // 4 * image_size // 4 * depth[1])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    def retrieve_feature(self, x):
        feature_map1 = F.relu(self.conv1(x))
        x = self.pool(feature_map1)
        feature_map2 = F.relu(self.conv2(x))
        return(feature_map1, feature_map2)
def rightness(y, target):
    pred = torch.max(y.data, 1)[1]
    rights = pred.eq(target.data.view_as(pred)).sum()
    out1 = y.size()[0]
    return(rights, out1)

## device change to GPU
net = ConvNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

record_t = []
record_v = []
weights = []

## Training start~
for epoch in range(num_epochs):
    train_rights = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        net.train()
        output = net(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        right = rightness(output, target)
        train_rights.append(right)
        if batch_idx % 100 == 0:
            net.eval()
            val_rights = []
            for (data_v, target) in validation_loader:
                data_v, target = Variable(data_v).cuda(), Variable(target).cuda()
                output = net(data_v)
                right = rightness(output, target)
                val_rights.append(right)
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            print(train_r)
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
            print('訓練週期:{} [{}/{} ({:.0f}%)]\t, Loss:{:.6f}\t, 訓練正確率:{:.2f}%\t, 驗證正確率:{:.2f}%'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100.* batch_idx/len(train_loader), loss.data, 100.* train_r[0]/train_r[1], 100.* val_r[0] / val_r[1]))
            record_t.append(100-100.*train_r[0]/train_r[1])
            record_v.append(100-100.*val_r[0]/val_r[1])
            weights.append([net.conv1.weight.data.clone(), net.conv1.bias.data.clone(), net.conv2.weight.data.clone(), net.conv2.bias.data.clone()])

## Testing start~
net.eval()
vals = []
for data, target in test_loader:
    data, target = Variable(data).cuda(), Variable(target).cuda()
    output = net(data)
    val = rightness(output, target)
    vals.append(val)

right = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
right_rate = 1.0*right[0]/right[1]
print(right_rate.data)

plt.figure(figsize=(10, 7))
# plt.plot(record)
xplot, = plt.plot(record_t)
yplot, = plt.plot(record_v)
plt.legend([xplot, yplot], ["train", "Validation"])
plt.xlabel('step')
plt.ylabel('Error rate')
plt.show()

net_cpu = net.cpu()
plt.figure(figsize=(10, 7))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(net_cpu.conv1.weight.data.numpy()[i, 0, ...])
plt.show()

## Analysis
# input_x = test_dataset[idx][0].unsqueeze(0)
# # print(input_x)
# # print(test_dataset[idx][0])
# feature_maps = net.retrieve_feature(Variable(input_x))
# plt.figure(figsize=(10, 7))
# for i in range(4):
#     plt.subplot(1, 4, i+1)
#     plt.imshow(feature_maps[0][0, i, ...].data.numpy())
# plt.show()
#
# plt.figure(figsize=(15, 10))
# for i in range(4):
#     for j in range(8):
#         plt.subplot(4, 8, i*8+j+1)
#         plt.imshow(net_cpu.conv2.weight.data.numpy()[j, i, ...])
#
# plt.figure(figsize=(10, 7))
# for i in range(8):
#     plt.subplot(2, 4, i+1)
#     plt.imshow(feature_maps[1][0, i, ...].data.numpy())
# plt.show()



