from io import BytesIO
import numpy as np
from PIL import Image
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from urllib.request import urlopen
from argparse import ArgumentParser
import sys
import flask

app = flask.Flask(__name__)
model = None
data_transforms = None

## build net & def function
image_size = 28
num_classes = 10
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

def load_data(cls_file):
    global classes
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
        # print(classes)

def load_transform():
    global data_transforms
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])


def load_model(path):
    global model
    model = torch.load(path)
    model.eval()

#### Flask app
## method = "POST"
@app.route("/predict", methods=["POST"])
def predict():
    output_dict = {"success": False}
    if flask.request.method == "POST":
        data = flask.request.json

        ## read the image in PIL format
        response = requests.get(data["image"])
        image = Image.open(BytesIO(response.content))
        # print(image)
        # print(type(image))

        ## transform image
        image_tensor = data_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        # print(image_tensor)

        ## prediction
        output = model(image_tensor)
        # print(output)
        _, predicted = torch.max(output.data, 1)
        print(classes[predicted])
        output_dict["predictions"] = classes[predicted]
        output_dict["success"] = True
    return flask.jsonify(output_dict), 200

## method = "GET"
@app.route("/performance", methods=["GET"])
def performance():
    output_dict = {"success": False}
    if flask.request.method == "GET":
        output_dict["performance"] = "XXX"
        output_dict["success"] = True
    return flask.jsonify(output_dict), 200

## main
if __name__ == "__main__":
    print(("* Loading pytorch model and Flask starting server... please wait until server has fully started"))
    load_model('integral_model.pth')
    load_data('D:\AI\PytorchBook_MNIST\classes.txt')
    load_transform()
    app.run(host="0.0.0.0", debug=True, port=5000)