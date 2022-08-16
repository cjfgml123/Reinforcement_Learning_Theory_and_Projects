from torch import nn
import torch.nn.functional as F
import Config

class DQN(nn.Module):
    def __init__(self,inputs,outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, outputs)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class DQN_v1(nn.Module):
    def __init__(self,inputs,outputs):
        super(DQN_v1, self).__init__()
        self.linear1 = nn.Linear(inputs, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.linear2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.linear4 = nn.Linear(64, outputs)
        
        self.relu = nn.LeakyReLU()
    
    def forward(self,x):
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.relu(self.bn3(self.linear3(x)))
        x = self.linear4(x)    
        return x 
    
