import torch
import torch.nn as nn
import torch.nn.functional as F

#######################################################################
# TODO: Design your own neural network
# You can define utility functions/classes here
#######################################################################

#######################################################################
# End of your code
#######################################################################


class MyNeuralNetwork(nn.Module):
    def __init__(self, do_batchnorm=False, p_dropout=0.0):
        super().__init__()
        self.do_batchnorm = do_batchnorm
        self.p_dropout = p_dropout


        self.conv11 = nn.Conv2d(3, 32, 3, padding= 1)
        if self.do_batchnorm:
          self.bn11 = nn.BatchNorm2d(32)
        self.gelu11 = nn.ReLU()
        
        self.conv12 = nn.Conv2d(32, 32, 3, padding= 1)
        if self.do_batchnorm:
          self.bn12 = nn.BatchNorm2d(32)
        self.gelu12 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2) 


        self.conv21 = nn.Conv2d(32, 64, 3, padding= 1)
        if self.do_batchnorm:
          self.bn21 = nn.BatchNorm2d(64)
        self.gelu21 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) 
        
        self.conv31 = nn.Conv2d(64, 128, 3, padding= 1)
        if self.do_batchnorm:
          self.bn31 = nn.BatchNorm2d(128)
        self.gelu31 = nn.ReLU()

        self.conv32 = nn.Conv2d(128, 128, 3, padding= 1)
        if self.do_batchnorm:
          self.bn32 = nn.BatchNorm2d(128)
        self.gelu32 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2) 

        self.fc1 = nn.Linear(2048, 256)
        self.relu4 = nn.ReLU()
        if p_dropout > 0.0:
          self.drop = nn.Dropout(p=p_dropout)
        
        self.fc2 = nn.Linear(256,100)
        self.relu5 = nn.ReLU()

        self.fc3 = nn.Linear(100 , 100)
        
    def forward(self, x):

        x = self.conv11(x)  
        if self.do_batchnorm:
            x = self.bn11(x)
        x = self.gelu11(x)
        
        x = self.conv12(x)  
        if self.do_batchnorm:
            x = self.bn12(x)
        x = self.gelu12(x)
        
        x = self.pool1(x)

        x = self.conv21(x)  
        
        if self.do_batchnorm:
            x = self.bn21(x)
        x = self.gelu21(x)
                
        x = self.pool2(x)
        
        x = self.conv31(x)  
        
        if self.do_batchnorm:
            x = self.bn31(x)
        x = self.gelu31(x)
        
        x = self.conv32(x)  
        if self.do_batchnorm:
            x = self.bn32(x)
        x = self.gelu32(x)
        
        x = self.pool3(x)

        x = torch.flatten(x, 1) 
        x = self.relu4(self.fc1(x))  
        
        if self.p_dropout > 0.0:
            x = self.drop(x)
        
        x = self.relu5(self.fc2(x))  
        if self.p_dropout > 0.0:
            x = self.drop(x)
        x = self.fc3(x)  
            
        return x