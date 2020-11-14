#Alec - Added this part because I wasn't apple to import. When modules are in parallel locations in folders this makes them visible to each other. 
#https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.append('../')

from input_output import load
import numpy as np
import statsmodels.api as sm
import pickle
import scipy.io as sio

# load the data
hrf = load.load_hrf_function()
# TODO different result compared to jupyter notebook, fMRI array and dataframe in jupyter do not match
fMRI = load.load_hrf(task='MOTOR')
task_paradigms = load.load_task_paradigms('MOTOR')

# do one hot encoding
task_paradigms_one_hot = load.separate_conditions(task_paradigms)

# do the convolution
task_paradigms_conv = load.do_convolution(task_paradigms_one_hot, hrf)

# fit the glm for every subject and region
print(f"Fitting GLM for {fMRI.shape[0]} subjects and {fMRI.shape[1]} regions...")
p_value = 0.05
activations = np.zeros((task_paradigms_one_hot.shape[0], fMRI.shape[1], task_paradigms_one_hot.shape[1] - 1))
betas = np.zeros((task_paradigms_one_hot.shape[0], fMRI.shape[1], task_paradigms_one_hot.shape[1] - 1))

for subject in range(fMRI.shape[0]):
    for region in range(fMRI.shape[1]):
        # dropping the first one hot encoding with 1: to circumvent dummy variable problem
        X = sm.add_constant(np.swapaxes(task_paradigms_conv[subject, 1:, :], 0, 1))
        y = fMRI[subject, region, :]
        mod = sm.OLS(y, X)
        res = mod.fit()
        if subject == 0 and region == 0:
            print(res.summary())
        p_values = res.pvalues[1:]
        coef = res.params[1:]
        # prints RuntimeError when y==0, i.e. all coefficients of the OLS are zero
        activations[subject, region, :] = p_values < p_value
        betas[subject, region, :] = coef

print("Done!")

# save the results
with open("activation.pickle", "wb") as f:
    pickle.dump(activations, f)

with open("beta.pickle", "wb") as f:
    pickle.dump(betas, f)

sio.savemat('betas.mat', {'beta': betas})


