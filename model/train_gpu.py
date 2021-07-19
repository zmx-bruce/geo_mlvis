#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib.request
import tarfile
from pathlib import Path
#from data import create_dir, download_data
import os


# In[5]:


args = {
    "batch_size": 3, # make this bigger if you are not running on binder
    "epochs": 2,
    "lr": 0.0001,
    "device": "cuda" # set to "cuda" if GPU is available
}


# In[21]:


import numpy as np
np.load(paths['x'][0]).shape


# In[22]:


from data import GlacierDataset
from torch.utils.data import DataLoader
path="/Users/app/Desktop/geo_mlvis/test/processed"
#path="/Volumes/KINGSTON/Remote Sensing Presentation/preprocessed/train"
paths = {
    "x": list(Path(path).glob("x*")),
    "y": list(Path(path).glob("y*"))
}

ds = GlacierDataset(paths["x"], paths["y"])
loader = DataLoader(ds, batch_size=args["batch_size"], shuffle=True)


# In[27]:


import torch.optim
from unet import Unet
from train import train_epoch

model = Unet(10, 3, 4, dropout=0.2).to(args["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
Loss_Total=[];Loss_Batch=[]
for epoch in range(args["epochs"]):
    train_epoch(model, loader, optimizer, args["device"], epoch)
    #Loss_Total.append(l[0])
    #Loss_Batch.append(l[1])
    
torch.save(model.state_dict(), "model.pt")


# In[ ]:


import pickle 
filename='loss.pkl'
with open(filename, 'wb') as f:  
    pickle.dump([Loss_Toral,Loss_Batch], f)

