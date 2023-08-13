
from typing import List
import torch
from collections import defaultdict, OrderedDict
import os
import sys
from argparse import ArgumentParser

from models import *

parser = ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--platform', type=str, required=True)
parser.add_argument('--output', type=str, default="data/merged_models/")
args = parser.parse_args()


if __name__ == '__main__':
    
    model = Clair3([1,1,1], platform=args.platform, add_indel_length=True)
    checkpoint = torch.load(args.model,map_location=torch.device('cpu'))
    state_dict = model.state_dict()
    # for k in checkpoint:
    #     print (k)
    
    if "encoder_weights" in checkpoint:
        pretrained_dict={}
        for k, v in checkpoint['encoder_weights'].items():
            if k[len("module."):] in state_dict:
                print(k)
                pretrained_dict[k[len("module."):]] = v
            elif k[len("_orig_mod."):] in state_dict:
                print(k)
                pretrained_dict[k[len("_orig_mod."):]] = v
            elif k in state_dict:
                pretrained_dict[k] = v
            else:
                print(k,"not exists")

        for k, v in checkpoint['classifier_weights'].items():
            if k[len("module."):] in state_dict:
                print(k)
                pretrained_dict[k[len("module."):]] = v
            elif k[len("_orig_mod."):] in state_dict:
                print(k)
                pretrained_dict[k[len("_orig_mod."):]] = v
            elif k in state_dict:
                pretrained_dict[k] = v
            else:
                print(k,"not exists")
    else:
        print("Not Implemented")
        assert(False)
        pretrained_dict = {k: v for k, v in checkpoint['encoder_weights'].items() if k in state_dict}

        for k, v in checkpoint['classifier_weights'].items():
            if k in state_dict:
                pretrained_dict[k] = v
            else:
                print(k,"not exists")
    
    # print(sys.getsizeof(pretrained_dict))
    print("Saved to {}".format(os.path.join(args.output,"merged.pt" )))
    state_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict)
    os.makedirs(os.path.join( args.output) , exist_ok=True)
    torch.save(  {
                'model_state_dict': model.state_dict(),
                } , os.path.join(args.output,"merged.pt" ) )