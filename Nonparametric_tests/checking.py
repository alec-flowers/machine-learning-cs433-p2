import scipy.io
import numpy as np

# TRUE BETAS: f'./Data/beta_truth/beta_tfMRI_MOTOR_LR_Glasser360.mat'
# THRESHOLDED BETAS CORRECTED: 'Nonparametric_tests/activations_results/thresholded_betas_dko/MOTOR_subj1_corr'
# THRESHOLDED BETAS UNCORRECTED: 'Nonparametric_tests/activations_results/thresholded_betas_dko/MOTOR_subj1_uncorr'
# TRUE BETAS THRESHOLDED: active_betas_MOTOR.mat

# for single subject!
TRUE_BETAS = f'../Data/beta_truth/beta_tfMRI_MOTOR_LR_Glasser360.mat' #Giulia
TRUE_BETAS_THRESHOLDED = '../GLM/betas/active_betas_MOTOR.mat'
THRESHOLDED_BETAS_CORRECTED = './activations_results/thresholded_betas_dko_MOTOR_subj1_corr.mat' #deep ko
THRESHOLDED_BETAS_UNCORRECTED = './activations_results/thresholded_betas_dko_MOTOR_subj1_uncorr.mat' #deep ko



true_betas = scipy.io.loadmat(TRUE_BETAS)['Beta'][:, :, 0]  # selecting subject 0
true_betas = np.swapaxes(true_betas, 0, 1) # getting data into the following format: [region, timeseries]
true_betas_thresholded = scipy.io.loadmat(TRUE_BETAS_THRESHOLDED)['beta'][0, :, :] # selecting subject 0
thresholded_betas_corrected = scipy.io.loadmat(THRESHOLDED_BETAS_CORRECTED)['beta']
thresholded_betas_uncorrected = scipy.io.loadmat(THRESHOLDED_BETAS_UNCORRECTED)['beta']


a = (true_betas_thresholded == thresholded_betas_uncorrected)
b = (true_betas_thresholded == thresholded_betas_corrected)

print('hi')
