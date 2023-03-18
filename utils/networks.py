import torch
import torch.nn as nn
import torchvision.models as models

class TabularNet(nn.Module):
    def __init__(self, num_features, net_mode = 0):
        super(TabularNet, self).__init__()
        self.mode = net_mode
        self.batchn1 = nn.BatchNorm1d(320)
        self.batchn2 = nn.BatchNorm1d(80)        
        
        self.fc1 = nn.Linear(num_features, 320)
        self.fc2 = nn.Linear(320, 80)        
        self.output = nn.Linear(80, 1)
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
        
        efficientnet = models.efficientnet_v2_s(pretrained=True)
        self.features = nn.Sequential(*list(efficientnet.children())[:-1]) #Output: 1280
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        if net_mode:
          self.tabular_net = TabularNet(num_features = 1280 , net_mode = 1)
        else:
          self.tabular_net = TabularNet(num_features = 1280)

    def forward(self, x):
        
        x = self.features(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.tabular_net(x)        
        return x

class EnsembleNet(nn.Module):
    def __init__(self, tabular_num_features):
        super(EnsembleNet, self).__init__()
        
        self.tabular_net = TabularNet(tabular_num_features) #Returns 80

        self.image_net = ImgNet() #Returns 80
        self.relu = nn.ReLU()
        self.batchn1 = nn.BatchNorm1d(160)        
        self.fc1 = nn.Linear(160, 40)
        self.batchn2 = nn.BatchNorm1d(40)        
        self.output = nn.Linear(40, 1)
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
