import urllib.request
import tarfile
from pathlib import Path
from data import create_dir, download_data
import os

# setup directory structure for download
data_dir = Path("/home/jovyan/data")
process_dir = data_dir / "processed"
create_dir(process_dir)

# download processed data
download_data(
    "https://uwmadison.box.com/shared/static/d54agxzb5g8ivr7hkac8nygqd6nrgrqr.gz", 
    process_dir / "train.tar.gz"
)

args = {
    "batch_size": 1, # make this bigger if you are not running on binder
    "epochs": 50,
    "lr": 0.0001,
    "device": "cpu" # set to "cuda" if GPU is available
}

from data import GlacierDataset
from torch.utils.data import DataLoader

paths = {
    "x": list((process_dir / "train").glob("x*")),
    "y": list((process_dir / "train").glob("y*"))
}

ds = GlacierDataset(paths["x"], paths["y"])
loader = DataLoader(ds, batch_size=args["batch_size"], shuffle=True)

import torch.optim
from unet import Unet
from train import train_epoch

model = Unet(10, 3, 4, dropout=0.2).to(args["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])

for epoch in range(args["epochs"]):
    train_epoch(model, loader, optimizer, args["device"], epoch)
    
torch.save(model.state_dict(), data_dir / "model.pt")

