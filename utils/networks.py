import torch
import torch.nn as nn
import torchvision.models as models

class TabularNet(nn.Module):
    def __init__(self, num_features, net_mode = 0):
        super(TabularNet, self).__init__()
        self.mode = net_mode
        self.batchn1 = nn.BatchNorm1d(128)
        self.batchn2 = nn.BatchNorm1d(32)        
        
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 32)        
        self.output = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):        
        x = self.fc1(x)
        x = self.batchn1(x)        
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.batchn2(x)        
        x = self.relu(x)
        x = self.dropout(x)

        if self.mode:
          x = self.output(x)
        return x

class ImgNet(nn.Module):
    def __init__(self, net_mode = 0):
        super(ImgNet, self).__init__()
        
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        if net_mode:
          self.tabular_net = TabularNet(num_features = 512 , net_mode = 1)
        else:
          self.tabular_net = TabularNet(num_features = 512)

    def forward(self, x):
        
        x = self.features(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.tabular_net(x)        
        return x

class EnsembleNet(nn.Module):
    def __init__(self, tabular_num_features):
        super(EnsembleNet, self).__init__()
        
        self.tabular_net = TabularNet(tabular_num_features) #Returns 64

        self.image_net = ImgNet() #Returns 64
        self.relu = nn.ReLU()
        self.batchn1 = nn.BatchNorm1d(64)        
        self.fc1 = nn.Linear(64, 32)
        self.batchn2 = nn.BatchNorm1d(32)        
        self.output = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x_tabular, x_image):
        x_tabular = self.tabular_net(x_tabular)
        x_image = self.image_net(x_image)
        x = torch.cat((x_tabular,x_image), dim=1)
        x = self.batchn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.batchn2(x)        
        x = self.relu(x)
        x = self.dropout(x)        
        x = self.output(x)
        return x