import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import util
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CRPS(nn.Module):
    def __init__(self, args):
        super(CRPS, self).__init__()
        self.K = 32
        self.args = args
        self.log_base = torch.FloatTensor([np.e]).to(DEVICE)
        assert (torch.log(self.log_base) - 1).abs() < 1e-4

    def I_ln(self, mu, scale, y, g):
        # integral of CDF^2 of lognormal times g;
        # using math from appendix A https://arxiv.org/pdf/1806.08324.pdf

        # X ~ N(mu, sigma) === > exp(X) ~ LogNormal(mu, sigma);
        # therefore, the Normal distribution we parameterize to compute the CDF uses the same sigma as scale
        norm = torch.distributions.normal.Normal(mu, scale) # CHECKED

        # approximation is as follows
        # let phi( x ) be the cdf of normal evaluated at x.
        # sum_k  0.5  * [ phi^2( log z_k) g(z_k) +  phi^2(log z_k-1) g(z_k-1) ] * [ z_k - z_k-1 ]

        # grid points to approximate the integral
        # creates K evenly spaced points from 1e-4 to 1
        grid_points = torch.tensor(np.linspace(1e-4, 1, self.K).astype(np.float32)).to(DEVICE)

        # compute z_k-1 and \phi^2( log z_k-1) for k = 1; so z_0 and phi(z_0)
        z_km1 = y*grid_points[0]
        phi_km1 = norm.cdf(z_km1.log()).view(-1)
        summand_km1 =  phi_km1.pow(2)*g(z_km1).view(-1)

        # return value
        retval = 0.0

        # loop over k from 1 to K-1, both included
        for k in range(1, self.K):
            z_k = y*grid_points[k]

            # compute phi^2(log z_k)
            phi_k = norm.cdf(z_k.log()).view(-1)
            summand_k =  phi_k.pow(2)*g(z_k).view(-1)

            # accumulate the summand 0.5 [ phi^2( log z_k) g(z_k) +  phi^2(log z_km1) g(z_km1) ] * [ z_k - z_km1 ]
            retval = retval + 0.5*(summand_k + summand_km1)*(z_k - z_km1)

            # update z_k-1 and phi^2(z_k-1)
            z_km1 = z_k
            summand_km1 = summand_k

        return retval

    def CRPS_surv_ln(self, mu, scale_lognormal, time, censor):
        # argument sigma = s.exp is the scale of the logNormal distribution
        Y = time
        I = lambda y: self.I_ln(mu, scale_lognormal, y, lambda y_: y_*0 + 1)
        I_ = lambda y: self.I_ln(-mu, scale_lognormal, 1/(y + 1e-4), lambda y_: (y_+1e-4).pow(-1))

        crps = I(Y) + (1 - censor) * I_(Y)
        return crps


    def I_cat(self, pred_params, tgts, part):
        # integral of CDF^2 of categorical if part = let
        # integral of (1 - CDF)^2 of categorical if part = right
        # using math from appendix A https://arxiv.org/pdf/1806.08324.pdf

        # therefore, the Normal distribution we parameterize to compute the CDF uses the same sigma as scale
        grid_points = torch.tensor(np.linspace(1e-4, 1, self.K).astype(np.float32)).to(DEVICE)

        def replace_y(tgts, grid_point):
            tgt_new = tgts.clone()
            tgt_new[:, 0] = tgt_new[:, 0]*grid_point
            return tgt_new

        # compute z_k-1 and \phi^2( log z_k-1) for k = 1; so z_0 and phi(z_0)
        tgt_km1 = replace_y(tgts, grid_point[0])
        z_km1 = tgt_km1[:, 0]

        assert False, 'DO THIS ASAP'

        phi_km1 = util.get_cdf_val(pred_params, tgt_km1, self.args, DEVICE)

        # return value
        retval = 0.0

        # loop over k from 1 to K-1, both included
        for k in range(1, self.K):
            z_k = y*grid_points[k]

            # compute phi^2(log z_k)
            phi_k = norm.cdf(z_k.log()).view(-1)
            summand_k =  phi_k.pow(2)*g(z_k).view(-1)

            # accumulate the summand 0.5 [ phi^2( log z_k) g(z_k) +  phi^2(log z_km1) g(z_km1) ] * [ z_k - z_km1 ]
            retval = retval + 0.5*(summand_k + summand_km1)*(z_k - z_km1)

            # update z_k-1 and phi^2(z_k-1)
            z_km1 = z_k
            summand_km1 = summand_k

        return retval

    def forward(self, pred_params, tgts, model_dist):
        tte, is_alive = tgts[:, 0], tgts[:, 1]
        tte = tte.to(DEVICE)

        assert model_dist == 'lognormal'

        '''
        if model_dist == 'cat':
            if self.args.interpolate:
                assert False, "NOT DONE"
                tte_max = self.args.bin_boundaries[-1]
                censor = is_alive
                left_loss = self.I_cat(pred_params, tgts, part='left')
                right_loss = self.I_cat(pred_params, tgts, part='right')

                loss = left_loss + (1-censor)*right_loss

            else:
                bin_boundaries = torch.Tensor(self.args.bin_boundaries).to(DEVICE)
                bin_len = bin_boundaries[1:] - bin_boundaries[:-1]
                bin_len = bin_len.unsqueeze(-1)
                is_alive = is_alive.to(DEVICE)

                cdf = torch.cumsum(torch.softmax(pred_params, dim=-1), dim=-1)
                times = tte.view(-1, 1)
                indices = torch.arange(pred_params.shape[1]).view(1, -1).to(DEVICE)
                mask = (times >= indices).float()
                # SCRPS RIGHT
                loss =  torch.mm(mask * (cdf**2), bin_len)\
                        + (1-is_alive) * torch.mm((1-mask)*((1 - cdf) ** 2), bin_len)
        '''

        mu,sigma = util.pred_params_to_lognormal_params(pred_params)
        scale_lognormal = sigma
        # what we use for CDF for dcal pred = torch.distributions.LogNormal(mu, scale_lognormal)
        loss = self.CRPS_surv_ln(mu, scale_lognormal, tte, is_alive)

        # Debugging numerical instability
        if torch.any(torch.isnan(loss)) or torch.any(loss == float('inf')):
            print("!!!!ERROR, tgts", tgts)
            pdb.set_trace()

        loss = loss.mean()
        return loss



