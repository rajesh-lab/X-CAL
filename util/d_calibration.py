import numpy as np
import torch

def d_calibration(points, is_alive, args, nbins=20, differentiable=False, gamma=1.0, device='cpu'):
    # each "point" in points is a time for datapoint i mapped through the model CDF
    # each such time_i is a survival time if not censored or a time sampled
    # uniformly in (censor time, max time
    # compute empirical cdf of cdf-mapped-times
    # Move censored points with cdf values greater than 1 - 1e-4 t0 uncensored group
    new_is_alive = is_alive.detach().clone()
    new_is_alive[points > 1. - 1e-4] = 0

    points = points.to(device).view(-1,1)
    # print(points[:200])
    # BIN DEFNITIONS
    # BIN DEFNITIONS
    # BIN DEFNITIONS
    # BIN DEFNITIONS
    # BIN DEFNITIONS
    # BIN DEFNITIONS
    bin_width = 1.0/nbins
    bin_indices = torch.arange(nbins).view(1,-1).float().to(device)
    bin_a = bin_indices * bin_width #+ 0.02*torch.rand(size=bin_indices.shape)
    noise = 1e-6/nbins*torch.rand(size=bin_indices.shape).to(device)
    if not differentiable:
        noise = noise * 0.
    cum_noise = torch.cumsum(noise, dim=1)
    bin_width = torch.tensor([bin_width]*nbins).to(device) + cum_noise
    bin_b = bin_a + bin_width

    bin_b_max = bin_b[:,-1]
    bin_b = bin_b/bin_b_max
    bin_a[:,1:] = bin_b[:,:-1]
    bin_width = bin_b - bin_a

    # CENSORED POINTS
    points_cens = points[new_is_alive.long()==1]
    upper_diff_for_soft_cens = bin_b - points_cens
    # To solve optimization issue, we change the first left bin boundary to be -1.;
    # we change the last right bin boundary to be 2.
    bin_b[:,-1] = 2.
    bin_a[:,0] = -1.
    lower_diff_cens = points_cens - bin_a # p - a
    upper_diff_cens = bin_b - points_cens # b - p
    diff_product_cens = lower_diff_cens * upper_diff_cens
    # NON-CENSORED POINTS

    if differentiable:
        # sigmoid(gamma*(p-a)*(b-p))
        bin_index_ohe = torch.sigmoid(gamma * diff_product_cens)
        exact_bins_next = torch.sigmoid(-gamma * lower_diff_cens)
    else:
        # (p-a)*(b-p)
        bin_index_ohe = (lower_diff_cens >= 0).float() * (upper_diff_cens > 0).float()
        exact_bins_next = (lower_diff_cens <= 0).float()  # all bins after correct bin

    EPS = 1e-13
    right_censored_interval_size = 1 - points_cens + EPS

    # each point's distance from its bin's upper limit
    upper_diff_within_bin = (upper_diff_for_soft_cens * bin_index_ohe)

    # assigns weights to each full bin that is larger than the point
    # full_bin_assigned_weight = exact_bins*bin_width
    # 1 / right_censored_interval_size is the density of the uniform over [F(c),1]
    full_bin_assigned_weight = (exact_bins_next*bin_width.view(1,-1)/right_censored_interval_size.view(-1,1)).sum(0)
    partial_bin_assigned_weight = (upper_diff_within_bin/right_censored_interval_size).sum(0)
    assert full_bin_assigned_weight.shape == partial_bin_assigned_weight.shape, (full_bin_assigned_weight.shape, partial_bin_assigned_weight.shape)

    # NON-CENSORED POINTS
    # NON-CENSORED POINTS
    # NON-CENSORED POINTS
    # NON-CENSORED POINTS
    # NON-CENSORED POINTS
    # NON-CENSORED POINTS
    points_uncens = points[new_is_alive.long() == 0]
    # compute p - a and b - p
    lower_diff = points_uncens - bin_a
    upper_diff = bin_b - points_uncens
    diff_product = lower_diff * upper_diff
    assert lower_diff.shape == upper_diff.shape, (lower_diff.shape, upper_diff.shape)
    assert lower_diff.shape == (points_uncens.shape[0], bin_a.shape[1])
    # NON-CENSORED POINTS

    if differentiable:
        # sigmoid(gamma*(p-a)*(b-p))
        soft_membership = torch.sigmoid(gamma*diff_product)
        fraction_in_bins = soft_membership.sum(0)
        # print('soft_membership', soft_membership)
    else:
        # (p-a)*(b-p)
        exact_membership = (lower_diff >= 0).float() * (upper_diff > 0).float()
        fraction_in_bins = exact_membership.sum(0)

    assert fraction_in_bins.shape == (nbins, ), fraction_in_bins.shape

    frac_in_bins = (fraction_in_bins + full_bin_assigned_weight + partial_bin_assigned_weight) /points.shape[0]
    return torch.pow(frac_in_bins - bin_width, 2).sum()

