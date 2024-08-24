import os
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim import SGD
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

def main():
    # Transforms
    transformer = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),  # 0-255 to 0-1, numpy to tensors
        transforms.Normalize([0.5, 0.5, 0.5],  # 0-1 to [-1,1] , formula (x-mean)/std
                             [0.5, 0.5, 0.5])
    ])

    # Path for training and testing directory
    train_path = 'E:\comsat\Comsats\Semester7\Image processing\ProjectBlackandwhite'

    train_loader = DataLoader(
        torchvision.datasets.ImageFolder(train_path, transform=transformer),
        batch_size=16, shuffle=True
    )
    # train_count = len(glob.glob(train_path + '/**/*.jpg'))

    # learning_rate_arr = [1, 0.1, 0.01, 1, 0.1, 0.01]
    # optimizer_arr = ['SGD', 'SGD', 'SGD', 'Adam', 'Adam', 'Adam']
    # batching_arr = [False, True, False, False, True, False]
    learning_rate_arr = [0.0001]
    optimizer_arr = ['Adam']
    batching_arr = [ True]
    for i in range(len(learning_rate_arr)):
        model = ConvNet(num_classes=2, batch=batching_arr[i])
        if optimizer_arr[i] == 'Adam':
            optimizer = Adam(model.parameters(), lr=learning_rate_arr[i])
        else:
            optimizer = SGD(model.parameters(), lr=learning_rate_arr[i])

        optimizer.zero_grad()
        loss_function = nn.CrossEntropyLoss()
        num_epochs = 5
        epoch_arr = [i for i in range(num_epochs)]
        train_acc_arr = []
        loss_arr = []

        for epoch in range(num_epochs):
            train_accuracy = 0.0
            train_loss = 0.0
            model.train()

            for images, labels in train_loader:

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                per_error, prediction = torch.max(outputs.data, 1)
                # print(labels.data)
                # print(prediction)
                # print((prediction == labels.data))
                # print(torch.sum(prediction == labels.data))
                train_accuracy += int(torch.sum(prediction == labels.data))
                # print(train_accuracy)
            # print(train_count)
            train_accuracy = train_accuracy / 326
            train_acc_arr.append(train_accuracy)

            loss_arr.append(train_loss)

            print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
                train_accuracy))
        torch.save(model, train_path+'\modelfirebatch16.pth')
        plt.plot(epoch_arr, train_acc_arr, label='Train Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        plt.plot(epoch_arr, loss_arr, label='Train error')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

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
