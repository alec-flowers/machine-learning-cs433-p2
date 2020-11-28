from input_output import load
import numpy as np
import pickle

PATH = '../GLM/betas/betas_knockoff_test.pickle'
PATH2 = '../GLM/betas/tvalues_knockoff_test.pickle'

with open(PATH, 'rb') as pickle_file:
    betas = pickle.load(pickle_file)

with open(PATH2, 'rb') as pickle_file:
    tvalues = pickle.load(pickle_file)


def threshold(real, ci_array):
    if (real <= ci_array[0]):
        return -1
    elif(real >= ci_array[1]):
        return 1
    return 0


def uncorrected_test(val, alpha=.05):
    '''
    Perform an uncorrected voxel level non-paramatric test on data of 1 subject.
    :param val: np.array of dimension (num knockoffs, regions, betas). index 0 is the true value
    :param alpha: Two sided threshold, 2*alpha is the test ex: alpha .05 is 90% level
    :return: np.array (region, thresholded beta) 1 for right side reject, -1 for left side reject, 0 accept
    '''
    region = val.shape[1]
    n = val.shape[0]
    ci = [int(n * alpha), int(n * (1 - alpha))]

    uncorrected_threshold = []
    # Loop over brain regions and at each region for each beta perform a hypothesis test
    for i in range(region):
        r = val[:, i, :]
        real = r[0, :]
        sort_r = np.sort(r, axis=0)
        ci_array = sort_r[ci, :]
        ind = []
        # Each beta compare against confidence interval and accept or reject H0
        for i in range(len(real)):
            ind.append(threshold(real[i], ci_array[:, i]))
        uncorrected_threshold.append(ind)
    return np.array(uncorrected_threshold)


def corrected_test(val, alpha=.05):
    '''
    Perform a single threshold image wise non-paramatric test on data of 1 subject.
    :param val: np.array of dimension (num knockoffs, regions, betas). index 0 is the true value
    :param alpha: Two sided threshold, 2*alpha is the test ex: alpha .05 is 90% level
    :return: np.array (region, thresholded beta) 1 for right side reject, -1 for left side reject, 0 accept
    '''
    num_beta = val.shape[2]
    regions = val.shape[1]
    ci = [int(regions * alpha), int(regions * (1 - alpha))]
    corrected_threshold = []
    # Loop over each beta and calculate maximal statistic for each beta over all brain regions.
    for i in range(num_beta):
        brain = val[:, :, i]
        real = brain[0, :]
        max_ = np.amax(brain, axis=0)
        image_beta = np.sort(max_)
        ci_array = image_beta[ci]
        ind = []
        # Compare beta values against maximal thresholded values and accept or reject null hypothesis.
        for r in real:
            ind.append(threshold(r, ci_array))
        corrected_threshold.append(ind)
    return np.array(corrected_threshold).T

corrected = corrected_test(betas)
uncorrected = uncorrected_test(betas)
load.save_pickle(corrected, 'act', 'act', "c_test")
load.save_pickle(uncorrected, 'act', 'act', "u_test")





