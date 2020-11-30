# Alec - Added this part because I wasn't apple to import. When modules are in parallel locations in folders this makes them visible to each other.
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.append('../')

from input_output import load
import numpy as np
import statsmodels.api as sm
import scipy.io as sio

def glm(fMRI, task_paradigms, hrf):
    """
    Computes the General Linear Model (GLM) from fMRI data to estimate the parameters for a given task

        Parameters:
        ----------
        fMRI: fMRI BOLD signal which is a 3-d array of size (n_subjects, n_regions, n_timepoints)
        task_paradigms: temporal details on the presentation of the tasks for each subject, with size (n_subjects, n_timepoints)
        hrf: Hemodynamic Response Function, used to convolute the task paradigm

        Return:
        ----------
        act: 2-d array of size (n_subjects, n_regions) with {0, 1} values corresponding to activation of
                    brain regions according to the result of the GLM
        betas: 2-d array of size (n_subjects, n_regions) with beta values resulting from GLM
    """
    assert fMRI.shape[2]==task_paradigms.shape[1], \
        f"fMRI and task_paradigms shapes do not match: {fMRI.shape[1]} and {task_paradigms.shape}"

    # do one hot encoding
    task_paradigms_one_hot = load.separate_conditions(task_paradigms)

    # do the convolution
    task_paradigms_conv = load.do_convolution(task_paradigms_one_hot, hrf)

    # fit the glm for every subject and region
    print(f"Fitting GLM for {fMRI.shape[0]} subjects and {fMRI.shape[1]} regions...")
    p_value = 0.05
    activations = np.zeros((task_paradigms_one_hot.shape[0], fMRI.shape[1], task_paradigms_one_hot.shape[1] - 1))
    betas = np.zeros((task_paradigms_one_hot.shape[0], fMRI.shape[1], task_paradigms_one_hot.shape[1] -1))
    tvalues = np.zeros((task_paradigms_one_hot.shape[0], fMRI.shape[1], task_paradigms_one_hot.shape[1] -1))

    for subject in range(fMRI.shape[0]):
        for region in range(fMRI.shape[1]):
            # dropping the first one hot encoding with 1: to circumvent dummy variable problem
            # X = sm.add_constant(np.swapaxes(task_paradigms_conv[subject, 1:, :], 0, 1))
            X = np.swapaxes(task_paradigms_conv[subject, 1:, :], 0, 1)
            y = fMRI[subject, region, :]
            mod = sm.OLS(y, X)
            res = mod.fit()
            if subject == 0 and region == 0:
                print(res.summary())
            p_values = res.pvalues
            coef = res.params
            tval = res.tvalues
            # prints RuntimeError when y==0, i.e. all coefficients of the OLS are zero
            activations[subject, region, :] = p_values < p_value
            betas[subject, region, :] = coef
            tvalues[subject, region, :] = tval

    print("Done!")

    return activations, betas, tvalues


