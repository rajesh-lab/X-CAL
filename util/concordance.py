import torch
import torch.nn.functional as F
import torch.utils.data as data
import util
import random
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from args import TestArgParser
from logger import TestLogger
from saver import ModelSaver
import numpy as np
from lifelines.utils import concordance_index
from evaluator import ModelEvaluator
import warnings


def concordance(args, test_loader, model):

    
    tte_per_batch = []
    is_alive_per_batch = []
    pred_time_per_batch = []
    

    for i, (src, tgt) in enumerate(test_loader):
        src=src.to(DEVICE)
        tgt=tgt.to(DEVICE)

        tte_per_batch.append(tgt[:,0])
        is_alive_per_batch.append(tgt[:,1])

        pred_params = model.forward(src.to(args.device))
        #if args.marginal_predict:
        #pred_params = torch.Tensor(marginal_counts).to(DEVICE).unsqueeze(0).repeat(pred_params.shape[0],1)
        pred = util.pred_params_to_dist(pred_params, args)
        pred_time = util.get_predict_time(pred,args) 
        pred_time_per_batch.append(pred_time)

    ttes = torch.cat(tte_per_batch).cpu().numpy()
    is_alives = torch.cat(is_alive_per_batch).long().cpu().numpy()
    pred_times = torch.cat(pred_time_per_batch).cpu().numpy()

   
    # Calculate Concordance
    concordance = concordance_index(ttes, pred_times, 1-is_alives)

    return concordance
