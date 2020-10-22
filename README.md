# X-CAL: Explicit Calibration for Survival Analysis

This repo contains the code for paper X-CAL: Explicit Calibration for Survival Analysis

This repo is derived from https://github.com/stanfordmlgroup/cdr-mimic

This repo works fine with python 3.7.4, pytorch 1.6.0, numpy 1.19.1, scipy 1.4.1 and lifeline 0.24.0.

```train.sh``` will run an example of training the categorical model on the gamma dataset. Change the variable ```lam``` in the script will train the model with different levels of X-CAL regularization.

After ```train.sh``` finishes, ```test.sh``` will give the results on the test dataset.