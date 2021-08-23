:
from pathlib import Path


args = {
    "device": "cpu", # set to "cuda" if gpu is available
    "out_dir": Path("predictions")
}

import torch
from unet import Unet

state = torch.load(Path("model.pt"), map_location=args["device"])
model = Unet(13, 3, 4).to(args["device"])
model.load_state_dict(state)
model = model.eval()

from data import GlacierDataset
from torch.utils.data import DataLoader

paths = {}
#for split in ["train", "test"]:
for split in ["train"]:
    paths[split] = {}
    for v in ["x", "y"]:
        paths[split][v] = list((Path("train").glob(v + "*"))
        paths[split][v].sort()

ds = {
    "train": GlacierDataset(paths["train"]["x"], paths["train"]["y"]),
    #"test": GlacierDataset(paths["test"]["x"], paths["test"]["y"])
}

from train import predictions

predictions(model, ds["train"], args["out_dir"] / "train", args["device"])
#predictions(model, ds["test"], args["out_dir"] / "test", args["device"])
