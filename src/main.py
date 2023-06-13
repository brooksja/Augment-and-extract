########## Packages ##########
import argparse
from pathlib import Path
import torch
import os
import gdown

from src.extract import extract_features
import src.extractors as extrs

########## Code ##########

def get_args():
    parser = argparse.ArgumentParser(prog='Augment-and-Extract',
                                        description = 'Program to augment images and then extract features from them.'
                                    )
    parser.add_argument('extractor',type=str,default='xiyue',choices=['xiyue','ozanciga'],help='Specify extractor model to use, default is xiyue')
#    parser.add_argument('checkpoint_path',type=Path,help='Specify path to extractor checkpoint')
    parser.add_argument('datadir',type=Path,help='Specify path to images to be augmented')
    parser.add_argument('outdir',type=Path,help='Specify path for output')
    parser.add_argument('-r','--repetitions',type=int,default=0,help='Specify number of augmentation repetitions to do, default 0')

    args = parser.parse_args()
    return args


args = get_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

match args.extractor:
    case 'xiyue':
        url = 'https://drive.google.com/drive/folders/1AhstAFVqtTqxeS9WlBpU41BV08LYFUnL'
        gdown.download_folder(url=url,quiet=True)
        extractor = extrs.load_xiyue('best_ckpt.pth').to(device)
    case 'ozanciga':
        url = 'https://github.com/ozanciga/self-supervised-histopathology/releases/download/tenpercent/tenpercent_resnet18.ckpt'
        gdown.download(url=url,quiet=True)
        extractor = extrs.load_ozanciga('tenpercent_resnet18.ckpt').to(device)
    case _:
        print('Error with extractor choice')

tile_paths = []
for root,dirs,_ in os.walk(args.datadir):
    for d in dirs:
        contents = [x for x in os.listdir(os.path.join(root,d)) if not x.startswith('.')]
        if len(contents) > 1:
            tile_paths.append(os.path.join(root,d))
#tile_paths = os.listdir(args.datadir)
#tile_paths = [os.path.join(args.datadir,p) for p in tile_paths if os.path.isdir(os.path.join(args.datadir,p))]
# = [Path(p.parent) for p in args.datadir.rglob('*') if p.is_file()]

extract_features(extractor=extractor,
                    tile_paths=tile_paths,
                    outdir=args.outdir,
                    augmentation_transforms=None, # TODO: make this an option in the argparser
                    repetitions=args.repetitions,
                )
