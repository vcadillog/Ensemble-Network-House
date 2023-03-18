import torch
import os
import numpy as np


from PIL import Image
from torch.utils.data import Dataset



class CustomDataset(Dataset):
    # def __init__(self, meta,img,y):
    def __init__(self, mode, meta = None, y = None,dir_path = None, transform = None):
      
        assert mode in ['meta','image','ensemble']
        self.mode = mode
        self.y = y
      
        if meta is not None and mode != 'image':
          self.meta = meta
        # self.img = img
        
        if dir_path is not None and mode !='meta':
          self.dir_path = dir_path
          self.features = ['frontal','kitchen', 'bedroom','bathroom']
          self.transform = transform
        
    def __len__(self):        
        return len(self.y)
        
    def __getitem__(self, idx):        
        
        if self.mode == 'meta':
          return self.meta[idx].astype(np.float32) , self.y.iloc[idx]
          
        else:
          img_idx = self.y.index[idx] + 1
          img_list = []          
          
          for feature in self.features:    
            img_path = os.path.join(self.dir_path,str(img_idx)+'_'+feature+'.jpg')            
            img = Image.open(img_path).convert("RGB")          
            if self.transform:
              img = self.transform(img)
            
            img_list.append(img)
          

          concat_img = torch.cat([torch.cat((img_list[0:2]),dim=1),torch.cat((img_list[2:4]),dim=1)],dim=2)        
          if self.mode == 'image':
            return concat_img , self.y.iloc[idx]
          else:
            return self.meta[idx].astype(np.float32), concat_img , self.y.iloc[idx]