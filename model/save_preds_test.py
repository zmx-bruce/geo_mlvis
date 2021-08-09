from pathlib import Path

data_dir = Path("/home/jovyan/data")
process_dir = data_dir / "processed"
args = {
    "device": "cpu", # set to "cuda" if gpu is available
    "out_dir": data_dir / "predictions"
}

from data import download_data

links = {
    "test_data": "https://uwmadison.box.com/shared/static/zs8vtmwbl92j5oq6ekzcfod11ym1w599.gz",
    "model": "https://uwmadison.box.com/shared/static/byb5lpny6rjr15zbx28o8liku8g6nga6.pt"
}

download_data(links["test_data"], process_dir / "test.tar.gz")
download_data(links["model"], data_dir / "model.pt", unzip = False)

import torch
from unet import Unet

state = torch.load(data_dir / "model.pt", map_location=args["device"])
model = Unet(13, 3, 4).to(args["device"])
model.load_state_dict(state)
model = model.eval()

from data import GlacierDataset
from torch.utils.data import DataLoader

paths = {}
for split in ["train", "test"]:
    paths[split] = {}
    for v in ["x", "y"]:
        paths[split][v] = list((process_dir / split).glob(v + "*"))
        paths[split][v].sort()

ds = {
    "train": GlacierDataset(paths["train"]["x"], paths["train"]["y"]),
    "test": GlacierDataset(paths["test"]["x"], paths["test"]["y"])
}

from train import predictions

predictions(model, ds["train"], args["out_dir"] / "train", args["device"])
predictions(model, ds["test"], args["out_dir"] / "test", args["device"])
