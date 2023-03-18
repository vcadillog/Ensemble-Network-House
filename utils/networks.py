import torch
import torch.nn as nn
import torchvision.models as models

class TabularNet(nn.Module):
    def __init__(self, num_features, net_mode = 0):
        super(TabularNet, self).__init__()
        self.mode = net_mode
        self.batchn1 = nn.BatchNorm1d(512)
        self.batchn2 = nn.BatchNorm1d(128)        
        
        self.fc1 = nn.Linear(num_features, 512)
        self.fc2 = nn.Linear(512, 128)        
        self.output = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.2)
        
        self.silu = nn.SiLU()

    def forward(self, x):        
        x = self.fc1(x)
        x = self.batchn1(x)        
        x = self.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batchn2(x)        
        x = self.silu(x)
        
        if self.mode:        
            x = self.dropout(x)
            x = self.output(x)
        return x

class ImgNet(nn.Module):
    def __init__(self, net_mode = 0):
        super(ImgNet, self).__init__()
        self.mode = net_mode
        efficientnet = models.efficientnet_v2_s(pretrained=True)
        self.features = nn.Sequential(*list(efficientnet.children())[:-1]) #Output: 1280
        
        if self.mode:
          self.tabular_net = TabularNet(num_features = 1280 , net_mode = 1)


    def forward(self, x):        
        x = self.features(x)        
        x = x.view(x.size(0), -1)
        if self.mode:
          x = self.tabular_net(x)        
        return x

class EnsembleNet(nn.Module):
    def __init__(self, tabular_num_features):
        super(EnsembleNet, self).__init__()
        
        self.tabular_net = TabularNet(tabular_num_features) #Returns 128

        self.image_net = ImgNet() #Returns 1280
   
        self.output = nn.Linear(1408, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x_tabular, x_image):
        x_tabular = self.tabular_net(x_tabular)
        x_image = self.image_net(x_image)
        x = torch.cat((x_tabular,x_image), dim=1)

        x = self.dropout(x)        
        x = self.output(x)
        return x
