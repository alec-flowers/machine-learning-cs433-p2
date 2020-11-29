import numpy as np
from input_output import load
from GLM import glm
from knockoff_af.knockoff_generator import fit_low_rank_knockoff, generate_knockoffs
# from deepknockoffs.examples.diagnostics import ScatterCovariance, compute_diagnostics
import matplotlib.pyplot as plt

# load the data
hrf = load.load_hrf_function()
fMRI = load.load_fmri(task='MOTOR')
tasks = load.load_task_paradigms(task='MOTOR')

# One subject for now
subject = 0
fmri = fMRI[subject]
tasks = np.expand_dims(tasks[subject, :], axis=0)

fit_knockoff = fit_low_rank_knockoff(fmri, rank=100)
num = 100
real_knockoffs = generate_knockoffs(fmri, fit_knockoff, num)

# ScatterCovariance(real_knockoffs[0, :, :].T, real_knockoffs[1, :, :].T)
# plt.show()
# plt.savefig("scatter_cov_lowrank_ko.pdf", format="pdf")

#compute_diagnostics(real_knockoffs[0, :, :].T, real_knockoffs[1, :, :].T, )

tasks = np.repeat(tasks, num+1, axis=0)

# Get beta values
activations, betas, tvalues = glm.glm(real_knockoffs, tasks, hrf)

# Save beta values
load.save_pickle(betas, 'knockoff_af/betas', 'knockoff_test_betas', 'MOTOR')
load.save_pickle(tvalues, 'knockoff_af/betas', 'knockoff_test_tvalues', 'MOTOR')
