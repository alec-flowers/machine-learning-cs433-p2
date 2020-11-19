#Alec - Added this part because I wasn't apple to import. When modules are in parallel locations in folders this makes them visible to each other. 
#https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
import sys
sys.path.append('../')

from input_output import load
import numpy as np
import statsmodels.api as sm
import pickle
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
        activations: 2-d array of size (n_subjects, n_regions) with {0, 1} values corresponding to activation of
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

    for subject in range(fMRI.shape[0]):
        for region in range(fMRI.shape[1]):
            # dropping the first one hot encoding with 1: to circumvent dummy variable problem
            #X = sm.add_constant(np.swapaxes(task_paradigms_conv[subject, 1:, :], 0, 1))
            X = np.swapaxes(task_paradigms_conv[subject, 1:, :], 0, 1)
            y = fMRI[subject, region, :]
            mod = sm.OLS(y, X)
            res = mod.fit()
            if subject == 0 and region == 0:
                print(res.summary())
            p_values = res.pvalues
            coef = res.params
            # prints RuntimeError when y==0, i.e. all coefficients of the OLS are zero
            activations[subject, region, :] = p_values < p_value
            betas[subject, region, :] = coef

    print("Done!")

    return activations, betas


def save_activations(activations, task):
    # save the results of the activations
    with open(f"./GLM/activations/activation_{task}.pickle", "wb") as f:
        pickle.dump(activations, f)


def save_betas(betas, task):
    # save the results of the beta values
    with open(f"./GLM/betas/betas_{task}.pickle", "wb") as f:
        pickle.dump(betas, f)


def save_betas_mat(betas, task):
    # save a beta file as .mat
    sio.savemat(f'./GLM/betas/betas_{task}.mat', {'beta': betas})


# def average_activations(activations):
betas_MOTOR = pickle.load(open('./GLM/betas/betas_MOTOR.pickle','rb'))
print(betas_MOTOR.shape) #(100,379,5)
# I want an average for all subjects, so I want (379,5)
avg = np.mean(betas_MOTOR, axis=0)
print(avg.shape)
sio.savemat(f'./GLM/betas/avg_MOTOR.mat', {'beta': avg})
