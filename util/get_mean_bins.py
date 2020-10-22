import numpy as np
import torch
import data
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_mean_bins(pred_params, mid_points):
    mid_points = torch.Tensor(mid_points).to(DEVICE)
    probs = torch.softmax(pred_params, dim=-1)
    mean_times = torch.mm(probs,(mid_points.unsqueeze(-1).float())).flatten()
    return mean_times
