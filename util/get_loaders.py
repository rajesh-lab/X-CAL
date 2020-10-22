
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import models
import optim
import util
from args import TrainArgParser

import pdb
import numpy as np
import random


from data import get_synthetic_loader




def get_train_loader(args,during_training=True):

    if during_training:
        TRAIN=True
        SHUFFLE=True
    else:
        TRAIN=False
        SHUFFLE=False

    # Get train loader
    if args.dataset=='synthetic':
        train_loader = get_synthetic_loader(args=args, is_training=TRAIN, 
                        phase='train',shuffle=SHUFFLE, dist=args.synthetic_dist,
                        censor=args.censor)
    return train_loader

 
def get_eval_loaders(during_training, args):
    if during_training==True:
        phase1='train'
        phase2='valid'
    else:
        phase1='valid'
        phase2='test'

    if args.dataset == 'synthetic':
        l1=get_synthetic_loader(args, is_training=False, phase=phase1,shuffle=False, 
                                dist=args.synthetic_dist,censor=args.censor)
        l2=get_synthetic_loader(args, is_training=False, phase=phase2,shuffle=False, 
                                dist=args.synthetic_dist,censor=args.censor)
        eval_loaders = [l1,l2]
    else:
        assert False, "Wrong Dataset Name"
    
    return eval_loaders


