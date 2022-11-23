import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    def __init__(self,z_dim=32):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(3, 12, 3, padding=1)  
        self.conv2 = nn.Conv2d(12,24, 3, padding=1)
        self.conv3 = nn.Conv2d(24,48, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(48 * 32 * 32, z_dim)#64->8
        self.fc2 = nn.Linear( z_dim , 48 * 32 * 32)
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(48, 24, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(24, 12, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(12, 3, 2, stride=2)
    
    def encode(self,x):
        ## encode ##
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)
    
    def decode(self,x):
        ## decode ##
        x = self.fc2(x)
        x = x.view(x.size(0),48,32,32)
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        return F.sigmoid(self.t_conv3(x))
    
    def forward(self, x):

        en = self.encode(x)
        de = self.decode(en)
        
        return de

class ConvNetwork(nn.Module):
    def __init__(self,z_dim=32):
        super(ConvNetwork, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(3, 12, 3, padding=1)  
        self.conv2 = nn.Conv2d(12,24, 3, padding=1)
        self.conv3 = nn.Conv2d(24,48, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(48 * 8 * 8, z_dim)
        self.fc2 = nn.Linear( z_dim , 48 * 8 * 8)
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(48, 24, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(24, 12, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(12, 3, 2, stride=2)
    
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)

class ConvNetwork_classification(nn.Module):
    def __init__(self,class_num=2):
        super(ConvNetwork_classification, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(3, 12, 3, padding=1)  
        self.conv2 = nn.Conv2d(12,24, 3, padding=1)
        self.conv3 = nn.Conv2d(24,48, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(48 * 32 * 32, class_num)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        return self.fc1(x)
        
