#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib.request
import tarfile
from pathlib import Path
from data import create_dir, download_data
import os
import pickle 

# In[5]:


args = {
    "batch_size": 3, # make this bigger if you are not running on binder #over 1400 patches
    "epochs": 50,
    "lr": 0.0001, # For SGD lr is higher, Adam,
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

model = Unet(10, 3, 4, dropout=0.2).to(args["device"])
print('model success')

optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])#Adam
print('optimize success')

Loss_Total=[];Loss_Batch=[]
for epoch in range(args["epochs"]):
    print('epoch success')
    l=train_epoch(model, loader, optimizer, args["device"], epoch)
    #L.append(l)
    Loss_Total.append(l[0])
    Loss_Batch.append(l[1])
    
torch.save(model.state_dict(),"model.pt")

# In[ ]:

filename='loss.pkl'
with open(filename, 'wb') as f:  
    pickle.dump([Loss_Total,Loss_Batch], f)
