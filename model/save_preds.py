#!/usr/bin/env python
# coding: utf-8

# ## Saving Predictions

# In[2]:


from pathlib import Path
args = {
    "device": "cuda", # set to "cuda" if gpu is available
    "out_dir":"predictions"
}


# In[4]:


import torch
from unet import Unet

state = torch.load(Path("model.pt"), map_location=args["device"])
model = Unet(10, 3, 4).to(args["device"])
model.load_state_dict(state)
model = model.eval()


# In[ ]:
path="npy"

from data import GlacierDataset
from torch.utils.data import DataLoader
paths = {}
#for split in ["train", "test"]:
for split in ["train"]:
    paths[split] = {}
    for v in ["x", "y"]:
       #paths[split][v] = list(Path(split+path).glob(v + "*"))
        paths[split][v] = list(Path("npy").glob(v + "*"))
        paths[split][v].sort()
ds = {
    "train": GlacierDataset(paths["train"]["x"], paths["train"]["y"]),
    #"test": GlacierDataset(paths["test"]["x"], paths["test"]["y"])
}


# In[ ]:


from train import predictions

predictions(model, ds["train"], Path(args["out_dir"]) / "train", args["device"])
#predictions(model, ds["test"], Path(args["out_dir"]) / "test", args["device"])


# In[ ]:




