from input_output import load
import numpy as np
import statsmodels.api as sm
import pickle

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
for subject in range(fMRI.shape[0]):
    for region in range(fMRI.shape[1]):
        # dropping the first one hot encoding with 1: to circumvent dummy variable problem and because
        # resting state of the experiment corresponds to no condition
        X = sm.add_constant(np.swapaxes(task_paradigms_conv[subject, 1:, :], 0, 1))
        y = fMRI[subject, region, :]
        mod = sm.OLS(y, X)
        res = mod.fit()
        if subject == 0 and region == 1:
            print(res.summary())
        p_values = res.pvalues[1:]
        tmp = p_values < p_value
        # prints RuntimeError when y==0, i.e. all coefficients of the OLS are zero
        activations[subject, region, :] = p_values < p_value
print("Done!")


# save the results
with open("activation.pickle", "wb") as f:
    pickle.dump(activations, f)

activation0 = activations[0,:,:]
np.savetxt('activation1.txt',activation0)
