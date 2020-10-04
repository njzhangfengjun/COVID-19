import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision

import numpy as np
import pandas as pd

from pathlib import Path

class VolumeDataset(Dataset):
    def __init__(self, df, cube_dir, input_channels):
        super(VolumeDataset, self).__init__()
        self._cube_dir = Path(cube_dir)

        if isinstance(df,str) or isinstance(df,Path):
            df = pd.read_csv(df)

        paths = df['path']
        labels = df['label']
        self._paths = list(paths)
        self._labels = list(labels)
        self._input_channels = input_channels
    

    def _load_cube(self, filepath):
        vol = np.load(filepath)
        vol = torch.tensor(vol)
        vol = vol.permute(3,0,1,2) # to (C,D,H,W)
        return vol


    def _get_cube(self, filepath):
        vol = self._load_cube(filepath)
        return vol


    def __len__(self):
        return len(self._paths)


    def __getitem__(self, idx):
        path = self._paths[idx]
        label = self._labels[idx]
        filepath = str(self._cube_dir/path)
        x = self._get_cube(filepath)
        y = label
        return x, y