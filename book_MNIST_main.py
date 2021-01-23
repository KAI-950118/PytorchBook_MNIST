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


net = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

## Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in net.state_dict():
#     print(param_tensor, "\t", net.state_dict()[param_tensor].size())
#
## Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])

## SAVE model "net" test
# torch.save(net.state_dict(), 'model.test')

## Training start~
record = []
weights = []
for epoch in range(num_epochs):
    train_rights = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
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
                data_v, target = Variable(data_v), Variable(target)
                output = net(data_v)
                right = rightness(output, target)
                val_rights.append(right)
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
            val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
            print('訓練週期:{} [{}/{} ({:.0f}%)]\t, Loss:{:.6f}\t, 訓練正確率:{:.2f}%\t, 驗證正確率:{:.2f}%'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100.* batch_idx/len(train_loader), loss.data, 100.* train_r[0]/train_r[1], 100.* val_r[0] / val_r[1]))
            record.append((100-100.*train_r[0]/train_r[1], 100-100.*val_r[0]/val_r[1]))
            weights.append([net.conv1.weight.data.clone(), net.conv1.bias.data.clone(), net.conv2.weight.data.clone(), net.conv2.bias.data.clone()])

# Print result of train
# plt.figure(figsize=(10, 7))
# plt.plot(record)
# plt.legend()
# plt.xlabel('step')
# plt.ylabel('Error rate')
# plt.show()

## Save trained model "net"
torch.save(net, 'integral_model.pth')
torch.save(net.state_dict(), 'model_only weights.pth')

## Testing start~
net.eval()
vals = []
for data, target in test_loader:
    data, target = Variable(data), Variable(target)
    output = net(data)
    val = rightness(output, target)
    vals.append(val)

right = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
right_rate = 1.0*right[0]/right[1]
print(right_rate.data)


## Loaded model test
model = torch.load('integral_model.pth')
model.eval()

vals_m = []
for data_m, target_m in test_loader:
    data_m, target_m = Variable(data_m), Variable(target_m)
    output_m = model(data_m)
    val_m = rightness(output_m, target_m)
    vals_m.append(val_m)

right_m = (sum([tup[0] for tup in vals_m]), sum([tup[1] for tup in vals_m]))
right_rate_m = 1.0*right_m[0]/right_m[1]
print(right_rate_m.data)


