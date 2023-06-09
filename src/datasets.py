########## Packages ##########
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np

########## Code ##########

# dataset for tiles
class TileDataset(Dataset):
    def __init__(self,tile_dir,transform=None,repetitions:int=1):
        self.tiles = list(tile_dir.glob('**/*.jpg'))
        if not self.tiles:
            raise NoTilesError(tile_dir)
        self.tiles *= repetitions
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self,idx):
        image = Image.open(self.tiles[idx])
        if self.transform:
            image = self.transform(image)

        return image,[]

# custom error for if there are no tiles in the specified directory
class NoTilesError(Exception):
    def __init__(self,tile_dir):
        print('No tiles found in {}'.format(tile_dir))
