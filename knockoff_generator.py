import numpy as np
from input_output import load
from GLM import glm
import fanok
from DeepKnockoffs import KnockoffMachine
from DeepKnockoffs import GaussianKnockoffs

# load the data
hrf = load.load_hrf_function()
fMRI = load.load_fmri(task='MOTOR')
tasks = load.load_task_paradigms(task='MOTOR')

# One subject for now
subject = 0
region = 0
fmri = fMRI[subject, :, :]
tasks = np.expand_dims(tasks[subject, :], axis=0)

# Initialize and fit low rank factor model for GaussianKnockoffs
factor_model = fanok.RandomizedLowRankFactorModel(rank=20)
knockoff = fanok.LowRankGaussianKnockoffs(factor_model)
knockoff.fit(X=fmri)

# Generate 100 knockoffs
iters = 100
knock_feat = np.zeros((iters, fmri.shape[0], fmri.shape[1]))
for i in range(iters):
    knock_feat[i, :, :] = knockoff.transform(X=fmri)

# Put data into correct shape for GLM
print(fmri.shape)
print(knock_feat.shape)
knock_feat = np.concatenate((np.expand_dims(fmri, axis=0), knock_feat), axis=0)
tasks = np.repeat(tasks, iters+1, axis=0)

# Get beta values
activations, betas = glm.glm(knock_feat, tasks, hrf)
glm.save_betas(betas, 'knockoff_test')


# selector = fanok.KnockoffSelector(fit_generator=False)
# p = first.shape[0]
# n = first.shape[1]
# model = 'test_knockoff'
# SigmaHat = np.cov(first, rowvar=False)
# w,v = np.linalg.eig(SigmaHat)
# w = np.where(w == 0, 1, w)
# SigmaHat_new = v @ np.diag(w) @ v.T
# def is_pos_def(x):
#     return np.all(np.linalg.eigvals(x) > 0)
#
# i = is_pos_def(SigmaHat_new)
#
# second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(first,0), method="sdp")
# corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)

# # Set the parameters for training deep knockoffs
# pars = dict()
# # Number of epochs
# pars['epochs'] = 100
# # Number of iterations over the full data per epoch
# pars['epoch_length'] = 100
# # Data type, either "continuous" or "binary"
# pars['family'] = "continuous"
# # Dimensions of the data
# pars['p'] = p
# # Size of the test set
# pars['test_size'] = 0
# # Batch size
# pars['batch_size'] = int(0.5*n)
# # Learning rate
# pars['lr'] = 0.01
# # When to decrease learning rate (unused when equal to number of epochs)
# pars['lr_milestones'] = [pars['epochs']]
# # Width of the network (number of layers is fixed to 6)
# pars['dim_h'] = int(10*p)
# # Penalty for the MMD distance
# pars['GAMMA'] = 1.0
# # Penalty encouraging second-order knockoffs
# pars['LAMBDA'] = 1.0
# # Decorrelation penalty hyperparameter
# pars['DELTA'] = 1.0
# # Target pairwise correlations between variables and knockoffs
# #pars['target_corr'] = corr_g
# # Kernel widths for the MMD measure (uniform weights)
# pars['alphas'] = [1.,2.,4.,8.,16.,32.,64.,128.]
#
# checkpoint_name = "tmp/" + model
# logs_name = 'tmp/' + model + "_progress.txt"
# machine = KnockoffMachine(pars, checkpoint_name=checkpoint_name, logs_name=logs_name)
#
# print("Fitting the knockoff machine...")
# machine.train(first)

