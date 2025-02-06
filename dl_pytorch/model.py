import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, do_batchnorm=False, p_dropout=0.0):
        super().__init__()
        self.do_batchnorm = do_batchnorm
        self.p_dropout = p_dropout

        self.conv1 = nn.Conv2d(3, 16, 3, padding= 1)
        if self.do_batchnorm:
          self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2) 

        self.conv2 = nn.Conv2d(16, 32, 3, padding= 1)
        if self.do_batchnorm:
          self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) 

        self.conv3 = nn.Conv2d(32, 64, 3, padding= 1, stride= 2)
        if self.do_batchnorm:
          self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(1024, 256)
        self.relu4 = nn.ReLU()
        if p_dropout > 0.0:
          self.drop = nn.Dropout(p=p_dropout)
        self.fc2 = nn.Linear(256, 100)

    def forward(self, x):
        # The shape of `x` is [bsz, 3, 32, 32]

        x = self.conv1(x)  # [bsz, 16, 32, 32]
        if self.do_batchnorm:
            x = self.bn1(x)
        x = self.pool1(self.relu1(x))  # [bsz, 16, 16, 16]

        x = self.conv2(x)  # [bsz, 32, 16, 16]
        if self.do_batchnorm:
            x = self.bn2(x)
        x = self.pool2(self.relu2(x))  # [bsz, 32, 8, 8]

        x = self.conv3(x)  # [bsz, 64, 4, 4]
        if self.do_batchnorm:
            x = self.bn3(x)
        x = self.relu3(x)

        x = torch.flatten(x, 1)  # [bsz, 1024]
        x = self.relu4(self.fc1(x))  # [bsz, 256]
        if self.p_dropout > 0.0:
            x = self.drop(x)
        x = self.fc2(x)  # [bsz, 100]
        return x
