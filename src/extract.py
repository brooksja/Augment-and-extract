########## Packages ##########
import os
import re
from pathlib import Path,Sequence
from re import T
from typing import Optional
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torchvision.transforms as T
import h5py
from tqdm import tqdm
import numpy as np

########## Code ##########

# dataset for tiles
class TileDataset(Dataset):
    def __init__(self,tile_dir,transform=None):
        self.tiles = list(tile_dir.glob('*.jpg'))
        if not self.tiles:
            raise NoTilesError(tile_dir)
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self,idx):
        image = Image.open(self.tiles[idx])
        if self.transform:
            image = self.transform(image)
        return image

# custom error for if there are no tiles in the specified directory
class NoTilesError(Exception):
    def __init__(self,tile_dir):
        print('No tiles found in {}'.format(tile_dir))

# function to extract coordinates from a filename
def get_coords(filename):
    if matches := re.match(r'.*\((\d+),(\d+)\)\.jpg', str(filename)):
        coords = tuple(map(int, matches.groups()))
        assert len(coords) == 2, 'Error extracting coordinates'
        return np.array(coords)
    else:
        return None

