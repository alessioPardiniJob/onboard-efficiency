import torch
import torch.nn as nn

from random import random as rnd
import numpy as np
import pandas as pd

from glob import glob
import os

from torchvision import transforms as T

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split




WIDTH = 128
HEIGHT = 128

class ReduceChannels(nn.Module):
    def __init__(self, in_channels=150, out_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=0.005, p=0.5):
        super().__init__()
        self.std = std
        self.mean = mean
        self.p = p

    def forward(self, img):
        if rnd() < self.p:
            noise = torch.randn_like(img) * self.std + self.mean
            return img + noise
        return img
    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets=None, transform=None, mean = None, std = None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.mean = mean
        self.std = std

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)  # Applichi qui la trasformazione
        if self.targets is not None:
            y = self.targets[idx]
            y = (y - self.mean) / self.std
            return x, y
        else:
            return x

    def __len__(self):
        return len(self.data)
 

train_transform = T.Compose([
    # ReduceChannels(),  # Resize allinea con Albumentations
    GaussianNoise(std=0.005, p=0.5),  # GaussNoise simulato
    T.RandomRotation(90),  # RandomRotate90
    T.RandomResizedCrop((WIDTH, HEIGHT), scale=(0.95, 1.05), ratio=(0.75, 1.33)),  # RandomResizedCrop
    T.RandomHorizontalFlip(p=0.5),  # Flip orizzontale casuale
    T.RandomVerticalFlip(p=0.5),  # Flip verticale casuale (equivalente a Flip generico)
    T.RandomAffine(degrees=90, translate=(0.05, 0.05)),  # ShiftScaleRotate (senza scaling)
])

reduce_transform = T.Compose([
    ReduceChannels()
])

resize_transform = T.Compose([
    T.Resize((WIDTH, HEIGHT))
])

resize_and_reduce_transform = T.Compose([
    ReduceChannels(),
    T.Resize((WIDTH, HEIGHT)),
])

def load_data(directory: str, tr = None):
    data = []
    sizes = set()
    all_files = np.array(
        sorted(
            glob(os.path.join(directory, "*.npz")),
            key=lambda x: int(os.path.basename(x).replace(".npz", "")),
        )
    )
    for file_name in all_files:
        with np.load(file_name) as npz:
            
            arr = npz['data']
            mask = npz["mask"]
            
            arr = torch.tensor(arr, dtype=torch.float32)
            mask = torch.tensor(~mask, dtype=torch.float32)
            
            arr = arr * mask
            
            if tr:
                arr = tr(arr)

        sizes.add((arr.shape[1], arr.shape[2]))
        data.append(arr)
    return data, sizes


def load_gt(file_path: str):
    gt_file = pd.read_csv(file_path)
    labels = gt_file[["P", "K", "Mg", "pH"]].values
    return labels

def get_dataloader(train_dataset, ds, config, batch_size, pin_memory=True):
    # Lazy import per evitare import circolare
    #import Utils.dp_utils as dp_utls

    if ds == "eurosat":
        num_train_samples = int(len(train_dataset) * (1 - config["tr_val_split"]))
        num_val_samples = len(train_dataset) - num_train_samples
        train_dataset, val_dataset = random_split(train_dataset, [num_train_samples, num_val_samples])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=1)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=1)

    elif ds == "hyperview":
        X_train, y_train = train_dataset
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=config["tr_val_split"], shuffle=True, random_state=config["seed"])

        mean = y_tr.mean(axis=0)
        std = y_tr.std(axis=0)

        train_dataset = CustomDataset(X_tr, y_tr, train_transform, mean, std)
        val_dataset   = CustomDataset(X_val, y_val, transform=None, mean=mean, std=std)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=1)
        val_dataloader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=1)

    return train_dataloader, val_dataloader
