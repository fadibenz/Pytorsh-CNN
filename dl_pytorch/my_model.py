import torch
import torch.nn as nn
import torch.nn.functional as F


class MyNeuralNetwork(nn.Module):
  def __init__(self, do_batchnorm=False, p_dropout=0.0):
        super().__init__()
        self.do_batchnorm = do_batchnorm
        self.p_dropout = p_dropout

        self.conv1 = nn.Conv2d(3, 32, 3, padding= 1)
        if self.do_batchnorm:
          self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2) 

        self.conv2 = nn.Conv2d(32, 64, 3, padding= 1)
        if self.do_batchnorm:
          self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) 

        self.conv3 = nn.Conv2d(64, 128, 3, padding= 1)
        if self.do_batchnorm:
          self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2) 

        self.fc1 = nn.Linear(2048, 356)
        if self.do_batchnorm:
          self.bn4 = nn.BatchNorm1d(356)
        self.relu4 = nn.ReLU()
        if p_dropout > 0.0:
          self.drop = nn.Dropout(p=p_dropout)
        self.fc2 = nn.Linear(356, 100)

  def forward(self, x):

        x = self.conv1(x)  
        if self.do_batchnorm:
            x = self.bn1(x)
        x = self.pool1(self.relu1(x))  

        x = self.conv2(x)  
        if self.do_batchnorm:
            x = self.bn2(x)
        x = self.pool2(self.relu2(x))  
        x = self.conv3(x)  
        if self.do_batchnorm:
            x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = torch.flatten(x, 1)  
        x = self.fc1(x)
        if self.do_batchnorm:
          x = self.bn4(x)
        
        x = self.relu4(x) 
        if self.p_dropout > 0.0:
            x = self.drop(x)
        x = self.fc2(x)  
        return x