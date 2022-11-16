########## Packages ##########
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
import numpy as np

########## Code ##########

# dataset for tiles
class TileDataset(Dataset):
    def __init__(self,tile_dir,transform=None,repetitions:int=1,tables={}):
        self.tiles = list(tile_dir.glob('**/*.jpg'))
        if not self.tiles:
            raise NoTilesError(tile_dir)
        self.tiles *= repetitions
        self.transform = transform
        if tables:
            clini = pd.read_excel(tables['clini']) if os.path.splitext(tables['clini'])[1]=='.xlsx' else pd.read_csv(tables['clini'])
            slide = pd.read_csv(tables['slide'])
            self.columns = tables['columns'].split(',')
            self.df = pd.merge(clini,slide,'left','PATIENT')

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self,idx):
        image = Image.open(self.tiles[idx])
        if self.transform:
            image = self.transform(image)

        if self.columns:
            # find the row based on filename, return values from desired columns
            f_id = os.path.basename(os.path.dirname(self.tiles[idx]))
            row = self.df[self.df['FILENAME'] == f_id]
            return image,tokenise(row[self.columns])
        else:
            return image,[]

# custom error for if there are no tiles in the specified directory
class NoTilesError(Exception):
    def __init__(self,tile_dir):
        print('No tiles found in {}'.format(tile_dir))

def tokenise(df):
#    if not (isinstance(df,pd.Series) or isinstance(df,pd.DataFrame)):
#        df = pd.Series(df)
    df = df.replace(['--',''],[np.nan,np.nan])
    df = df.replace(' ','_')

    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer,df))

    data = [torch.tensor(vocab(tokenizer(item)),dtype=torch.long) for item in df]
    return data