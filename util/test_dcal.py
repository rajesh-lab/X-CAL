import numpy as np
import torch
import random
from d_calibration import d_calibration
SEED = 1
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if use_cuda else "cpu")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
N = 1000000
points = torch.rand(N).to(device)
is_alive = torch.zeros(N).to(device)
# mask = torch.rand(N)
# is_alive = (points > mask).long()
# print('amount of censoring', torch.mean(is_alive.float()))
# points[points > mask] = mask[points > mask]
# is_alive = ((points > 0.5)&(mask>0.5)).long()
# points[(points > 0.5)&(mask>0.5)] = 0.5


class FakeArgs():
    def __init__(self):
        self.interpolate = False

args = FakeArgs()

d_cal = d_calibration(points, is_alive, args,nbins=5, differentiable=False, gamma=1.0, device=device)
print('N', N)
print('d_cal', d_cal)
