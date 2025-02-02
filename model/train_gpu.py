#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib.request
import tarfile
from pathlib import Path
from data import create_dir, download_data
import os
import pickle 
import sys

# In[5]:

#optimizer_type=str(sys.argv[1])
#l_rate=float(sys.argv[2])

args = {
    "batch_size": 1, # make this bigger if you are not running on binder #over 1400 patches
    "epochs": 50,
    "lr": 0.001, # For SGD lr is higher, Adam,
    "device": "cuda" # set to "cuda" if GPU is available
}


# In[22]:



from data import GlacierDataset
from torch.utils.data import DataLoader

#path=Path('npy/test/processed')
path=Path("npy")
paths = {
    "x": list(path.glob("x*")),
    "y": list(path.glob("y*"))
}
print('paths success')
ds = GlacierDataset(paths["x"], paths["y"])
print('ds success')
loader = DataLoader(ds, batch_size=args["batch_size"], shuffle=True)
print('loader success')
# In[27]:


import torch.optim
from unet import Unet
from train import train_epoch
print('load packages success')

model = Unet(13, 3, 4, dropout=0.1).to(args["device"])# decrease the drop out. download the ndvi, ndwi, 
print('model success')
#if optimizer_type=='Adam':
optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])#Adam
#if optimizer_type=='SGD':
    #optimizer = torch.optim.SGD(model.parameters(), lr=args["lr"])#SGD
#optimizer_type Adam or SGD
print('optimize success')

Loss_Total=[];Loss_Batch=[]
for epoch in range(args["epochs"]):
    print('epoch success')
    l=train_epoch(model, loader, optimizer, args["device"], epoch)
    #L.append(l)
    Loss_Total.append(l[0])
    Loss_Batch.append(l[1])
    
#torch.save(model.state_dict(),f"model_{optimizer_type}_{l_rate}.pt")
torch.save(model.state_dict(),f"model.pt")


# In[ ]:

#filename=f'loss_{optimizer_type}_{l_rate}.pkl'
filename=f'loss.pkl'
with open(filename, 'wb') as f:  
    pickle.dump([Loss_Total,Loss_Batch], f)
