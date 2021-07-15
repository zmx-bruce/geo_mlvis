#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import urllib.request
import tarfile
from pathlib import Path
import os


# In[ ]:


args = {
    "batch_size": 10, # make this bigger if you are not running on binder
    "epochs": 5,
    "lr": 0.0001,
    "device": "cpu" # set to "cuda" if GPU is available
}


# In[ ]:


from data import GlacierDataset
from torch.utils.data import DataLoader

paths = {
    "x": list(Path("npy/test/processed").glob("x*")),
    "y": list(Path("npy/test/processed").glob("y*"))
}

ds = GlacierDataset(paths["x"], paths["y"])
loader = DataLoader(ds, batch_size=args["batch_size"], shuffle=True)


# In[ ]:


import torch.optim
from unet import Unet
from train import train_epoch

model = Unet(10, 3, 4, dropout=0.2).to(args["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
Loss_Total=[];Loss_Batch=[]
for epoch in range(args["epochs"]):
    l=train_epoch(model, loader, optimizer, args["device"], epoch)
    Loss_Total.append(l[0])
    Loss_Batch.append(l[1])
    
torch.save(model.state_dict(), "model.pt")


# In[ ]:


import pickle 
filename='loss.pkl'
with open(filename, 'wb') as f:  
    pickle.dump([Loss_Toral,Loss_Batch], f)

