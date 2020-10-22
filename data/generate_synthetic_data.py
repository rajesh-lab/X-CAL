import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable

import pdb

#import synthetic_loader 

import argparse


parser = argparse.ArgumentParser(description='Data Gen')

parser.add_argument('--D', type=int, default=32)
parser.add_argument('--dist', type=str, default='lognormal', choices=['gamma', 'lognormal'])
parser.add_argument('--N', type=int, default=100000, help='number of train samples')

args = parser.parse_args()

Uniform = torch.distributions.Uniform
Beta = torch.distributions.Beta
MVN = torch.distributions.MultivariateNormal
Normal = torch.distributions.Normal
Bernoulli = torch.distributions.Bernoulli
Weibull = torch.distributions.Weibull
Gamma = torch.distributions.Gamma
StudentT = torch.distributions.StudentT
RELU = torch.nn.functional.relu
RELU6 = torch.nn.functional.relu6
grad = torch.autograd.grad
kl = torch.distributions.kl_divergence
EPS = 1e-4

def bad(a):
    return torch.any(torch.isnan(a))
def random_function(x, out_sz=2):

    with torch.no_grad(): 
        D = x.size()[1]
        layer_one = nn.Linear(D, out_sz)
        # init layer to have small params! important to avoid huge times!
        torch.nn.init.uniform_(layer_one.weight, a=-0.1, b=0.1)
        torch.nn.init.uniform_(layer_one.bias, a=-0.1, b=0.1)
        w = layer_one.weight.data
        b = layer_one.bias.data
        z = layer_one(x)
    return z, w, b

def x_to_gamma_dist(x, hidden_sz=32):
    '''
    creates lognormal data, mean function of x
    '''
    
    z,w,b = random_function(x, out_sz = 2)
    z1,z2 = torch.chunk(z,2,dim=-1)

    mean = z1.exp()

    var = 0.001
    print("VARIANCES ARE HARD CODED TO BE", var)
    '''
    convert from mean,var to alpha,beta
    see https://stats.stackexchange.com/questions/342639/how-to-find-alpha-and-beta-from-a-gamma-distribution
    '''
    alpha = mean.pow(2) / var
    beta = mean / var
    
    
    # alpha is shape
    # beta is rate, which is 1 / scale
    dist = Gamma(alpha,beta)

    return dist, w, b

def x_to_lognormal_dist(x, hidden_sz=32):
    '''
    creates lognormal data, mean function of x, fixed variance
    '''
    z,w,b = random_function(x, out_sz = 1)
    loc = z
    loc= loc.detach()
    C = 0.1
    print("VARIANCES ARE HARD CODED TO BE", C)

    def scale_from_loc(loc, C):
        return (((1+(1 + ((4*C) / (2*loc).exp()) ).sqrt() ) / 2).log()).sqrt()
    scale = scale_from_loc(loc, C)
    # scale = root ( log [ ( 1+ sqrt(1+4C/exp(2mu))) / 2 ] )
    dist = LogNormal(loc, scale)
    return dist, w, b, C



train_N = args.N
valid_N = args.N//2
test_N = args.N//2
hidden_sz = 32

print("dist is", args.dist)

N = train_N + valid_N + test_N



# also try 0.1, 1.0, and 5.0 instead of 10.0 for MVN covariance
x_dist = MVN(loc=torch.zeros(args.D),covariance_matrix=10.0*torch.eye(args.D))
#x_dist = StudentT(df=3.0*torch.ones(args.D), loc=torch.zeros(args.D), scale=torch.ones(args.D))
x = x_dist.sample(sample_shape=(N,))

# CHOOSE DATA GENERATING DIST
if args.dist == 'gamma':
    p_surv_t,wsurv,bsurv = x_to_gamma_dist(x)
    p_censor_t,wcens,bcens = x_to_gamma_dist(x)
elif args.dist == 'lognormal':
    p_surv_t,wsurv,bsurv,csurv = x_to_lognormal_dist(x)
    p_censor_t,wcens,bcens,ccens = x_to_lognormal_dist(x)    
else:
    assert False

# TIMES ARE SAMPLED, NOT DETERMINISTICALLY THE MEAN
surv_t = p_surv_t.sample()
censor_t = p_censor_t.sample()

# TRAIN VALID TEST SPLIT
VAL_SPLIT = train_N
TEST_SPLIT = train_N + valid_N

x_tr = x[: VAL_SPLIT]
x_va = x[VAL_SPLIT : TEST_SPLIT]
x_te = x[TEST_SPLIT : ]

surv_t_tr = surv_t[: VAL_SPLIT]
surv_t_va = surv_t[VAL_SPLIT : TEST_SPLIT]
surv_t_te = surv_t[TEST_SPLIT : ]

censor_t_tr = censor_t[: VAL_SPLIT]
censor_t_va = censor_t[VAL_SPLIT : TEST_SPLIT]
censor_t_te = censor_t[TEST_SPLIT : ]

if args.dist=='lognormal': 
    # SAVE LOC AND SCALE PARAMETERS
    loc_tr = p_surv_t.loc[ : VAL_SPLIT]
    scale_tr = p_surv_t.scale[ : VAL_SPLIT]
    loc_va = p_surv_t.loc[VAL_SPLIT :  TEST_SPLIT]
    scale_va = p_surv_t.scale[VAL_SPLIT : TEST_SPLIT]
    loc_te = p_surv_t.loc[TEST_SPLIT : ]
    scale_te = p_surv_t.scale[TEST_SPLIT : ]

    dist_tr = torch.distributions.LogNormal(loc_tr, scale_tr)
    dist_va = torch.distributions.LogNormal(loc_va, scale_va)
    dist_te = torch.distributions.LogNormal(loc_te, scale_te)

    torch.save(loc_tr,args.dist+'_loc_tr.pt')
    torch.save(scale_tr,args.dist+'_scale_tr.pt')
    torch.save(loc_va,args.dist+'_loc_va.pt')
    torch.save(scale_va,args.dist+'_scale_va.pt')
    torch.save(loc_te,args.dist+'_loc_te.pt')
    torch.save(scale_te,args.dist+'_scale_te.pt')
elif args.dist == 'gamma':
    # SAVE ALPHA AND BETA PARAMETERS
    alpha_tr = p_surv_t.concentration[ : VAL_SPLIT]
    beta_tr = p_surv_t.rate[ : VAL_SPLIT]
    alpha_va = p_surv_t.concentration[ VAL_SPLIT : TEST_SPLIT]
    beta_va = p_surv_t.rate[ VAL_SPLIT : TEST_SPLIT]
    alpha_te = p_surv_t.concentration[ TEST_SPLIT : ]
    beta_te = p_surv_t.rate[ TEST_SPLIT : ]
    
    dist_tr = torch.distributions.Gamma(alpha_tr, beta_tr)
    dist_va = torch.distributions.Gamma(alpha_va, beta_va)
    dist_te = torch.distributions.Gamma(alpha_te, beta_te)


    torch.save(alpha_tr, args.dist+'_alpha_tr.pt')
    torch.save(beta_tr, args.dist+'_beta_tr.pt')
    torch.save(alpha_va, args.dist+'_alpha_va.pt')
    torch.save(beta_va, args.dist+'_beta_va.pt')    
    torch.save(alpha_te, args.dist+'_alpha_te.pt')
    torch.save(beta_te, args.dist+'_beta_te.pt')
else:
    assert False, "invalid dist"

# COMPARE MIN/MAX OF MEANS TO THE VARIANCE
print("dist train min mean", dist_tr.mean.min())
print("dist train max mean", dist_tr.mean.max())

print("dist valid min mean", dist_va.mean.min())
print("dist valid max mean", dist_va.mean.max())

print("dist test min mean", dist_te.mean.min())
print("dist test max mean", dist_te.mean.max())


# CHECK TRUE LOGP
logp_tr = dist_tr.log_prob(surv_t_tr).mean()
logp_va = dist_va.log_prob(surv_t_va).mean()
logp_te = dist_te.log_prob(surv_t_te).mean()
print("logp tr:", logp_tr)
print("logp va:", logp_va)
print("logp te:", logp_te)

# params for linear function of x that makes params t|x
torch.save(wsurv, args.dist+'_wsurv.pt')
torch.save(bsurv, args.dist+'_bsurv.pt')
torch.save(wcens, args.dist+'_wcens.pt')
torch.save(bcens, args.dist+'_bcens.pt')

# data
torch.save(x_tr, args.dist+'_x_tr.pt')
torch.save(x_va, args.dist+'_x_va.pt')
torch.save(x_te, args.dist+'_x_te.pt')
torch.save(surv_t_tr, args.dist+'_surv_t_tr.pt')
torch.save(surv_t_va, args.dist+'_surv_t_va.pt')
torch.save(surv_t_te, args.dist+'_surv_t_te.pt')
torch.save(censor_t_tr, args.dist+'_censor_t_tr.pt')
torch.save(censor_t_va, args.dist+'_censor_t_va.pt')
torch.save(censor_t_te, args.dist+'_censor_t_te.pt')
