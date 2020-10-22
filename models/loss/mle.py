import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import sys
import util
sys.path.append("...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_log(x, EPS=1e-4):
    return (x + EPS).log()

class MLE(nn.Module):
    def __init__(self, args):
        super(MLE, self).__init__()
        self.log_base = torch.FloatTensor([np.e]).to(DEVICE)
        self.eps = 1e-4
        self.args = args

    # Fast forward
    def forward(self, pred_params, tgt, model_dist):
        # note if you use chunk instead
        # these keep an extra dim and cause
        # broadcasting problems 
        # dont need ratio here, ignoring 3rd dim
        tte, is_alive = tgt[:, 0], tgt[:, 1]


        is_alive = is_alive.long()
        cdf = util.get_cdf_val(pred_params, tgt, self.args)
        logpdf = util.get_logpdf_val(pred_params, tgt, self.args)
        
        survival_func = 1.0 - cdf
        log_survival = safe_log(survival_func, EPS=self.eps)
       

        def bad(x):
            return torch.any(torch.isnan(x)) or torch.any(torch.isinf(x))

        if bad(log_survival):
            print("BAD LOG SURVIVAL in MLE")
        if bad(logpdf):
            print("BAD LOG PDF in MLE")
            
        loglikelihood = (1 - is_alive) * logpdf + is_alive * log_survival
       
        nll = -1.0 * loglikelihood
        return nll.mean(dim=-1)
