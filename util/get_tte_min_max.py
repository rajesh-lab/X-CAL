import numpy as np
import torch
import data


def get_tte_min_max(args):

    train_loader = util.get_train_loader(args)
    val_loader, test_loader = util.get_eval_loaders(during_train=False, args=args)

    # Get min and max for time
    tte_min = torch.Tensor([10000.])
    tte_max = torch.Tensor([0.])


    for src, tgt, extra_surv_t, extra_censor_t in train_loader:
        tte = tgt[:, 0]
        batch_tte_min = tte.min()
        batch_tte_max = tte.max()
        if batch_tte_max > tte_max:
            tte_max = batch_tte_max
        if batch_tte_min < tte_min:
            tte_min = batch_tte_min
    for src, tgt, extra_surv_t, extra_censor_t in val_loader:
        tte = tgt[:, 0]
        batch_tte_min = tte.min()
        batch_tte_max = tte.max()
        if batch_tte_max > tte_max:
            tte_max = batch_tte_max
        if batch_tte_min > tte_min:
            tte_min = batch_tte_min
    for src, tgt, extra_surv_t, extra_censor_t in test_loader:
        tte = tgt[:, 0]
        batch_tte_min = tte.min()
        batch_tte_max = tte.max()
        if batch_tte_max > tte_max:
            tte_max = batch_tte_max
        if batch_tte_min > tte_min:
            tte_min = batch_tte_min
    tte_min = tte_min.item()
    tte_max = tte_max.item()
    return tte_min, tte_max

