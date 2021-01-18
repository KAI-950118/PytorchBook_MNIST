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

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

app = flask.Flask(__name__)
model = None
# data_transforms = None


# class VGG16_model(nn.Module):
#     def __init__(self, numClasses=7):
#         super(VGG16_model, self).__init__()
#         self.vgg16 = nn.Sequential(
#             nn.Conv2d(3, 64, 3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, 3),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, 3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, 3),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 256, 3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 512, 3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, 3),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, numClasses)
#         )
#
#     def forward(self, x):
#         x = self.vgg16(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

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

# def load_transform():
#     global data_transforms
#     data_transforms = transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])


def load_model(path):
    global model
    model = torch.load(path)
    # model.eval()


@app.route("/predict", methods=["POST"])
def predict():
    output_dict = {"success": False}
    if flask.request.method == "POST":
        data = flask.request.json

        # read the image in PIL format
        image = requests.get(data["image"])
        # image = Image.open(BytesIO(response.content))

        # transform image
        # image_tensor = data_transforms(image).float()
        # image_tensor = image_tensor.unsqueeze_(0).to(device)

        # predict and max
    #     output = model(image)
    #     _, predicted = torch.max(output.data, 1)
    #     output_dict["predictions"] = classes[predicted]
    #     output_dict["success"] = True
    # return flask.jsonify(output_dict), 200
    return 123


@app.route("/performance", methods=["GET"])
def performance():
    output_dict = {"success": False}
    if flask.request.method == "GET":
        output_dict["performance"] = "XXX"
        output_dict["success"] = True
    return flask.jsonify(output_dict), 200


if __name__ == "__main__":
    # parser = ArgumentParser()
    # # parser.print_help()
    # # parser.add_argument('--eval', help='evaluation log', type=str, default=None)
    # parser.add_argument('--model', help='D:\AI\PytorchBook_MNIST\model_only weights.pth', type=str, default=None)
    # parser.add_argument('--port', help='port', type=int, default=5000)
    # parser.add_argument('--classes', help='D:\AI\PytorchBook_MNIST\classes.txt', type=str, default=None)
    # args = parser.parse_args()
    print(("* Loading pytorch model and Flask starting server... please wait until server has fully started"))
    # if (args.model is None):
    #     print("You have to load the model by --model")
    #     sys.exit()
    # if (args.classes is None):
    #     print("You have to load the classes file by --classes")
    #     sys.exit()
    # if (args.eval is None):
    #     print("You have to load the evaluation log by --eval")
    #     sys.exit()

    load_model('model_only weights.pth')
    load_data('D:\AI\PytorchBook_MNIST\classes.txt')
    # load_transform()
    app.run(host="0.0.0.0", debug=True, port=5000)