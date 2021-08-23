import urllib.request
import tarfile
from pathlib import Path
import os

# setup directory structure for download
data_dir = Path("/home/jovyan/data")
process_dir = data_dir / "processed"
create_dir(process_dir)


args = {
    "batch_size": 1, # make this bigger if you are not running on binder
    "epochs": 50,
    "lr": 0.0001,
    "device": "cuda" # set to "cuda" if GPU is available
}

from data import GlacierDataset
from torch.utils.data import DataLoader

paths = {
    "x": list((Path("npy")).glob("x*")),
    "y": list((Path("npy")).glob("y*"))
}

ds = GlacierDataset(paths["x"], paths["y"])
loader = DataLoader(ds, batch_size=args["batch_size"], shuffle=True)

import torch.optim
from unet import Unet
from train import train_epoch

model = Unet(13, 3, 4, dropout=0.2).to(args["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])

L=[]
for epoch in range(args["epochs"]):
    l=train_epoch(model, loader, optimizer, args["device"], epoch)
    L.append(l)
    
torch.save(model.state_dict(), "model.pt")
