import numpy as np
import torch
import random
import util

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bad(tensor):
    return torch.any(torch.isnan(tensor))

class CatDist():
    def __init__(self, pred_params,args):
        self.pred_params = pred_params
        self.args = args
        self.interpolate = args.interpolate

    def predict_time(self):
        pred_time = util.get_mean_bins(self.pred_params, self.args.mid_points)
        return pred_time

    def log_prob(self,times):
        times=times.long()
        log_unnormalized_probs = self.pred_params[torch.arange(self.pred_params.shape[0]),times]
        normalizer = torch.logsumexp(self.pred_params, dim=-1)
        return log_unnormalized_probs - normalizer
    
    def cdf(self,times, ratio):
        times=times.long()
        params = self.pred_params
        batch_sz = params.size()[0]
        K = params.size()[-1]
        times = times.view(-1,1)
        indices = torch.arange(K).view(1,-1).to(DEVICE)
        
        '''
        Linear Interpolation for CDF
        '''
        # compute some masks
        # 1's up to but not including correct bin, then zeros
        mask1 = (times > indices).float()
        # 1 up to and including correct bin, then zeros
        mask2 = (times >= indices).float()

        all_probs = torch.softmax(params, dim=-1)
        cdf_km1 = (all_probs * mask1).sum(dim=-1) 
        prob_k = all_probs[range(batch_sz), times.squeeze()]

        cdf_k = (all_probs * mask2).sum(dim=-1)
        assert torch.all((cdf_k - (cdf_km1 + prob_k)).abs() < 1e-4)
        
        if not self.interpolate:
            return cdf_k
        else:
            '''
            define cdf_i(k) = sum_{j=0}^{k} prob_i(j)

            linear_interpolation for bin k
            a_k, b_k, t_i     bin(t_i) = k   if  a_k < t_i < b_k
            
            without interpolation
            cdf_i(bin(t_i)) = cdf_i(k)
            
            with interpolation:
            ratio_ik = (t_i - a_k) / (b_k - a_k)  
            cdf_i(t_i) = cdf_i(k-1) + prob_i(k) * ratio_ik
            '''
            
            
            probs = cdf_km1 + prob_k * ratio

            if torch.any(probs > 1.0+1e-4):
                print('*'*39)
                print('Warning: cdf is greater than one!')
                print('probs', probs[probs > 1.+1e-7])
                print('ratio', ratio[probs > 1.+1e-7])
                print('*'*39)
            

            
            
            if bad(probs):
                print('probs is nan', bad(probs))

            interpolated_cdf = probs
            return interpolated_cdf


def cat_l2_norm(params):
    diffs =params[:, 1:] - params[:, :-1]
    total_diffs= torch.norm(diffs)/params.shape[0]
    return total_diffs



def get_bin_for_time(tte,bin_boundaries):
    # we exclude bin_boundaries[0] so that the below sum equals 0
    # for a datapoint in the 0th bin.

    # we exclude bin_boundaries[-1] because otherwise a time larger
    # than all bin boundaries would have a sum thats K instead of K-1
    
    boundaries_to_consider = bin_boundaries[1:-1].view(1, -1)
    # The bin intex tte is in
    tte_cat = (tte > boundaries_to_consider).sum(dim=-1)
    return tte_cat


def cat_bin_target(args,tgt,bin_boundaries):
    

    bin_boundaries = torch.tensor(bin_boundaries).to(DEVICE)
    tte, is_alive = tgt[:, 0].unsqueeze(-1), tgt[:, 1].unsqueeze(-1)
   
    batch_sz = tte.size()[0]

    tte_cat = get_bin_for_time(tte, bin_boundaries) 
    

    tte = tte.squeeze()

    # there are K+1 bin boundaries
    K = (bin_boundaries[:-1]).size()[0]
    lower_boundaries = bin_boundaries[:-1].view(1, -1)
    upper_boundaries = bin_boundaries[1:].view(1, -1)
    assert lower_boundaries.size()[-1] == K
    
    # BATCH_SZ x K
    tte_as = lower_boundaries.repeat(batch_sz, 1)[range(batch_sz),tte_cat]
    tte_bs = upper_boundaries.repeat(batch_sz, 1)[range(batch_sz),tte_cat]

    
    ratio = (tte- tte_as) / (tte_bs - tte_as)
    ratio[tte > bin_boundaries[-1]] = 1.0
    ratio[tte < bin_boundaries[0]] = 0.0
    ratio = ratio.unsqueeze(-1)
    tte_cat = tte_cat.float().unsqueeze(-1)
    

    if not args.interpolate:
        '''
        OVERWRITING TO COMPUTE NON CENSORED DCAL
        ON CENSORED POINTS THAT ARE IN MAX BIN
        '''
        max_bin = args.num_cat_bins - 1.
        not_max_bin = (tte_cat < max_bin).long()
        is_alive = (is_alive * not_max_bin)

    tgt = torch.cat((tte_cat, is_alive, ratio), dim=-1)
    tgt=tgt.to(DEVICE)
    return tgt




