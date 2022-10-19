########## Packages ##########
import argparse
from pathlib import Path
import torch

from src.extract import extract_features
import src.extractors as extrs

########## Code ##########

def get_args():
    parser = argparse.ArgumentParser(prog='Augment-and-Extract',
                                        description = 'Program to augment images and then extract features from them.'
                                    )
    parser.add_argument('extractor',type=str,default='xiyue',choices=['xiyue','ozanciga'],help='Specify extractor model to use, default is xiyue',dest='extractor')
    parser.add_argument('checkpoint_path',type=Path,required=True,help='Specify path to extractor checkpoint',dest='checkpoint_path')
    parser.add_argument('datadir',type=Path,required=True,help='Specify path to images to be augmented',dest='datadir')
    parser.add_argument('outdir',type=Path,required=True,help='Specify path for output',dest='outdir')
    parser.add_argument(['-r','--repetitions'],type=int,default=1,help='Specify number of augmentation repetitions to do, default 1',dest='reps')

    args = parser.parse_args()
    return args


args = get_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

match args.extractor:
    case 'xiyue':
        extractor = extrs.load_xiyue(args.checkpoint_path).to(device)
    case 'ozanciga':
        extractor = extrs.load_ozanciga(args.checkpoint_path).to(device)
    case _:
        print('Error with extractor choice')

extract_features(extractor=extractor,
                    tile_paths=args.datadir,
                    outdir=args.outdir,
                    augmentation_transforms=None, # TODO: make this an option in the argparser
                    repetitions=args.reps
                )
