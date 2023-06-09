import random
import torch
import numpy as np
import wandb
import torch.nn as nn

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader,random_split
from dataset import CustomDataset
from networks import EnsembleNet, ImgNet,TabularNet
from torchvision.transforms import ToTensor, Compose, Resize,Normalize


def data_transforms(size=100,visualize=False):
  if visualize:
    return Compose([                 
                  Resize([size,size]),                          
                  ToTensor()                                    
                  ])   
  else:    
    return Compose([                 
                  Resize([size,size]),                          
                  ToTensor(),                  
                  Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                  ])

def set_seed(seed):
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)

def epoch_train(mode,model,device,optimizer,criterion,train_loader):
    model.train()
    train_loss = []
    if mode == 'ensemble':      
      for tmp_meta,tmp_img, tmp_y in train_loader:

          tmp_meta,tmp_img,tmp_y = tmp_meta.to(device) , tmp_img.to(device) , tmp_y.float().view(-1,1).to(device)

          optimizer.zero_grad()              
          outputs = model(tmp_meta,tmp_img)          
          loss = criterion(outputs, tmp_y)
          loss.backward()
          optimizer.step()
          loss_np = loss.detach().cpu().numpy()
          train_loss.append(loss_np)        
    elif mode == 'meta':      
      for tmp_meta, tmp_y in train_loader:

          tmp_meta,tmp_y = tmp_meta.to(device) , tmp_y.float().view(-1,1).to(device)

          optimizer.zero_grad()              
          outputs = model(tmp_meta)          
          loss = criterion(outputs, tmp_y)
          loss.backward()
          optimizer.step()          
          loss_np = loss.detach().cpu().numpy()
          train_loss.append(loss_np)          
    else:      
      for tmp_img, tmp_y in train_loader:

          tmp_img,tmp_y = tmp_img.to(device) , tmp_y.float().view(-1,1).to(device)

          optimizer.zero_grad()              
          outputs = model(tmp_img)          
          loss = criterion(outputs, tmp_y)
          loss.backward()
          optimizer.step()
          loss_np = loss.detach().cpu().numpy()
          train_loss.append(loss_np)        

    return model,optimizer,train_loss

def eval_model(mode,model,device,criterion,val_loader):
    model.eval()
    val_loss = []
    if mode == 'ensemble':
      with torch.no_grad():
          for tmp_meta,tmp_img, tmp_y in val_loader:

              tmp_meta,tmp_img,tmp_y = tmp_meta.to(device) , tmp_img.to(device) , tmp_y.float().view(-1,1).to(device)

              outputs = model(tmp_meta,tmp_img)
              
              loss = criterion(outputs, tmp_y)
              val_loss.append(loss.detach().cpu().numpy())
    elif mode == 'meta':
      with torch.no_grad():
          for tmp_meta,tmp_y in val_loader:

              tmp_meta,tmp_y = tmp_meta.to(device), tmp_y.float().view(-1,1).to(device)

              outputs = model(tmp_meta)              
              loss = criterion(outputs, tmp_y)              
              val_loss.append(loss.detach().cpu().numpy())
    else:
      with torch.no_grad():
          for tmp_img, tmp_y in val_loader:

              tmp_img,tmp_y = tmp_img.to(device) , tmp_y.float().view(-1,1).to(device)

              outputs = model(tmp_img)              
              loss = criterion(outputs, tmp_y)
              val_loss.append(loss.detach().cpu().numpy())
                          
    return val_loss

def reload_model(mode,X,y,size=100):
  if mode == 'ensemble':
    dataset = CustomDataset(mode=mode,meta = X,dir_path = './Houses-dataset/Houses Dataset', y = y, transform= data_transforms(size))
    model = EnsembleNet(tabular_num_features=X.shape[1])
  elif mode == 'meta':
    dataset = CustomDataset(mode=mode,meta = X, y = y)
    model = TabularNet(num_features=X.shape[1]  , net_mode = 1)
  else:
    dataset = CustomDataset(mode=mode,dir_path = './Houses-dataset/Houses Dataset', y = y, transform= data_transforms(size))
    model = ImgNet( net_mode = 1)  
  return model, dataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def fold_training(mode, model,device,num_folds,criterion,seed, dataset,batch_size,num_epochs,fold = None,train_idx = None,val_idx = None):
  model = model.to(device)      
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4 , weight_decay=0.1)    

  g = torch.Generator()
  g.manual_seed(seed)

  if num_folds>1:
    print(f'Fold {fold + 1}')
  # Define the training and validation data loaders for the current fold
    train_data = torch.utils.data.Subset(dataset, train_idx)
    val_data = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True , worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_data, batch_size=batch_size,shuffle=True, worker_init_fn=seed_worker, generator=g)

  else:
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = dataset_size - train_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True , worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_data, batch_size=batch_size,shuffle=True, worker_init_fn=seed_worker, generator=g)     
  # Initialize variables for logging and tracking best model
  best_loss = float('inf')
  best_model = None

  # Train the model for the current fold

  for epoch in range(num_epochs):


      model,optimizer,train_loss  = epoch_train(mode,model,device,optimizer,criterion,train_loader)
      val_loss = eval_model(mode,model,device,criterion,val_loader)             

      train_loss = np.mean(train_loss)
      val_loss = np.mean(val_loss)
      if num_folds > 1:
        wandb.log({'Train loss': train_loss,
                  'Validation Loss': val_loss,
                  'Epoch': epoch+1,
                  'Fold': fold+1
                  }) 

      else:
        wandb.log({'Train loss': train_loss,
                  'Validation Loss': val_loss,
                  'Epoch': epoch+1                  
                  }) 


      print(f'Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')
      # Save best model
      if val_loss < best_loss:
          best_loss = val_loss              
          best_model = model.state_dict()


  return best_loss, best_model


def run(mode, X,y, num_epochs = 20, batch_size = 32, num_folds = 5,img_size = 100):
  
  assert mode in ['meta','image','ensemble']
  
  
  # Define the number of epochs and number of folds for cross validation
  seed = 42
  num_folds = num_folds
  batch_size = batch_size  
  criterion = nn.MSELoss()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # Initialize Weight and Biases
  wandb.init(project='House')  

  # Perform cross validation
  
  
  model , dataset = reload_model(mode,X,y,img_size)

  best_fold_loss = []
  best_fold_model = []

  
  if num_folds>1:
    kf = KFold(n_splits=num_folds)  
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
      
      best_loss, best_model = fold_training(mode, model,device,num_folds,criterion,seed, dataset,batch_size,num_epochs,fold,train_idx ,val_idx)                             
                              
      best_fold_loss.append(best_loss)
      best_fold_model.append(best_model)
    
    best_idx = np.argmin(best_fold_loss)  
    best_model = best_fold_model[best_idx]
    torch.save(best_model, f'best_kfold_{num_folds}_{mode}_model.pt')            


  else: 
    _, best_model = fold_training(mode, model,device,num_folds,criterion,seed, dataset,batch_size,num_epochs)                              
    torch.save(best_model, f'best_{mode}_model.pt')    
