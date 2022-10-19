########## Packages ##########
import torch
import torch.nn as nn
import torchvision.models as models
from src.RetCLL import ResNet

########## Code ##########

def load_xiyue(checkpoint_path):
    """Prepares pre-trained RetCLL model. Based on Marugoto

    Args:
        checkpoint_path:  Path to the model checkpoint file.  Can be downloaded
            from <https://drive.google.com/drive/folders/1AhstAFVqtTqxeS9WlBpU41BV08LYFUnL>.
    """
    model = ResNet.resnet50(num_classes=128,mlp=False, two_branch=False, normlinear=True)
    weights = torch.load(checkpoint_path)
    model.fc = nn.Identity()
    model.load_state_dict(weights, strict=True)
    model.name = 'xiyue_wang'

    return model

def load_ozanciga(checkpoint_path):
    """Prepares resnet18 model with ozanciga weights. Based on Marugoto

    Args:
        checkpoint_path:  Path to the model checkpoint file.  Can be downloaded
            from <https://github.com/ozanciga/self-supervised-histopathology/releases/tag/tenpercent>.
    """
    model = models.resnet18(pretrained=False)
    weights = torch.load(checkpoint_path)
    model.fc = nn.Identity()
    model.load_state_dict(weights,strict=True)
    model.name = 'ozanciga'

    return model