########## Packages ##########
import os
import re
from pathlib import Path
from typing import Optional,Sequence
import torch
from torch.utils.data import ConcatDataset,DataLoader
import torchvision.transforms as T
import h5py
from tqdm import tqdm
import numpy as np
import json


from src.datasets import TileDataset

########## Code ##########

# function to extract coordinates from a filename
def get_coords(filename):
    if matches := re.match(r'.*\((\d+),(\d+)\)\.jpg', str(filename)):
        coords = tuple(map(int, matches.groups()))
        assert len(coords) == 2, 'Error extracting coordinates'
        return np.array(coords)
    else:
        return None

def extract_features(extractor,tile_paths:Sequence[Path],outdir:Path,augmentation_transforms=None,repetitions:Optional[int]=1,tables = {}):

    # define default transforms to use
    default_augmentation = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(p=.5),
        T.RandomVerticalFlip(p=.5),
        T.RandomApply([T.GaussianBlur(3)], p=.5),
        T.RandomApply([T.ColorJitter(
            brightness=.1, contrast=.2, saturation=.25, hue=.125)], p=.5),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    normal_transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # if no augmentation transforms are specified, use default
    if not augmentation_transforms:
        augmentation_transforms = default_augmentation
    
    # set and create output directory
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    # write settings into .json file for reference
    with open(outdir/'info.json','w') as f:
        json.dump({'extractor':extractor.name,
                    'augmentation':str(augmentation_transforms),
                    'repetitions':repetitions},
                    f)
    
    for tile_path in tqdm(tile_paths):
        tile_path = Path(tile_path)
        fname = 
        # check if h5 for slide already exists / slide_tile_path path contains tiles
        if (h5outpath := outdir/f'{tile_path.name}.h5').exists():
            print(f'{h5outpath} already exists.  Skipping...')
            continue
        if not next(tile_path.glob('**/*.jpg'), False):
            print(f'No tiles in {tile_path}.  Skipping...')
            continue

        extractor = extractor.eval()

        for i in range(repetitions+1):
            if i == 0:
                data = TileDataset(tile_path,normal_transform,1,tables)
                aug=False
            else:
                data = TileDataset(tile_path,augmentation_transforms,1,tables)
                aug=True
                h5outpath = outdir/f'{tile_path.name}_aug_{i}.h5'
            dl = DataLoader(data,batch_size=64,shuffle=False,num_workers=int(os.cpu_count()/2),drop_last=False)

            feats = torch.tensor([])
            # extract features
            for batch in tqdm(dl,leave=False):
                img,extras = batch
                new_feats = extractor(img.type_as(next(extractor.parameters()))).cpu().detach()
                if tables:
                    extras = torch.t(torch.stack(extras).squeeze())
                    if extras.dim() == 1:
                        extras = torch.unsqueeze(extras,0)
                    new_feats = torch.concat((new_feats,extras),dim=1)
                feats = torch.concat((feats,new_feats),dim=0)

            # write tile coords, features, etc to h5 file
            with h5py.File(h5outpath,'w') as f:
                f['coords'] = [get_coords(fn) for fn in data.tiles]
                f['feats'] = feats.cpu().numpy()
                f['augmented'] = np.repeat(aug,len(data))
                assert len(f['feats']) == len(f['augmented'])
                f.attrs['extractor'] = extractor.name
    print('Augmentation and extraction complete')


