import torch
import os
import torch.nn as nn
from PIL import Image
from torch.optim import Adam
from torchvision.transforms import transforms
import torchvision
import matplotlib.pyplot as plt

def main():
    prediction=['fire','notfire']
    model = ConvNet(num_classes=2, batch=True)
    test_path = 'E:\comsat\Comsats\Semester7\Image processing\Project(FA20-BCS-079)\Model'
    model = torch.load(test_path+'\modelfire.pth')
    # with open(train_path+'\modelfire.pth','rb') as f:
    #     model.load_state_dict(torch.load(f))
    img = Image.open('E:\comsat\Comsats\Semester7\Image processing\Project(FA20-BCS-079)\DataSet\TestDataBlackAndWhite\ilenametest0.png').resize((200,200))
    img = img.convert('RGB')
    img.show()
    image_tensor = transforms.ToTensor()(img).unsqueeze(0)
        # Apply the transformation to convert the image to a tensor

    print(image_tensor.shape)
    print(prediction[torch.argmax(model(image_tensor))],'prediction')


class ConvNet(nn.Module):
    def __init__(self, num_classes=2, batch=False):
        super(ConvNet, self).__init__()
        self.isBatch = batch
        # Input shape= (32,3,200,200)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Shape= (32,12,200,200)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Shape= (32,12,200,200)
        self.relu1 = nn.ReLU()
        # Shape= (32,12,200,200)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=23, kernel_size=3, stride=1, padding=1)
        # Shape= (32,23,200,200)
        self.relu2 = nn.ReLU()
        # Shape= (32,23,200,200)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # Shape= (32,23,100,100)

        self.conv3 = nn.Conv2d(in_channels=23, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Shape= (32,32,100,100)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        # Shape= (32,32,100,100)
        self.relu3 = nn.ReLU()
        # Shape= (32,32,100,100)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Shape= (32,20,100,100)
        self.bn3 = nn.BatchNorm2d(num_features=20)
        # Shape= (32,20,100,100)
        self.relu4 = nn.ReLU()
        # Shape= (32,20,100,100)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # Shape= (32,20,50,50)

        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(in_features=50 * 50 * 20, out_features=500)
        # self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=500, out_features=100)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=100, out_features=num_classes)

        # Feed forwad function
    def forward(self, input):
        output = self.conv1(input)
        if self.isBatch:
            output = self.bn1(output)
        output = self.relu1(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.pool1(output)

        output = self.conv3(output)
        if self.isBatch:
            output = self.bn2(output)
        output = self.relu3(output)

        output = self.conv4(output)
        output = self.relu4(output)

        output = self.pool2(output)

        output = self.fl(output)
        output = self.fc1(output)
        # output = self.dropout1(output)
        output = self.fc2(output)
        # output = self.dropout2(output)
        output = self.fc3(output)
        return output


main()
