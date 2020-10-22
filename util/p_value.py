from scipy.stats import chi2

def get_p_value(d_cal, degree_of_freedom, num_of_samples, num_of_bins):
    # d_cal: the value of d_calibration
    # degree_of_freedom: degree of freedom for chisq dist
    # num_of_samples: number of samples used to calculate dcal
    # num_of_bins: number of bins in dcalibration

    # Test statistic = dcal*num_of_samples*number_of_bins
    test_statistic = d_cal * num_of_samples * num_of_bins
    # Test distribution Chisq
    rv = chi2(degree_of_freedom)
    # p value = 1-rv.cdf(test_statistic)
    p_value = 1. - rv.cdf(test_statistic)
    return test_statistic, p_value



if __name__ == '__main__':
    p_value = get_p_value(0.08, 19, 10000, 20)
    print('p_value', p_value)
