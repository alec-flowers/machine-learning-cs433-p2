from implementation import load
import numpy as np
import pickle
import scipy.io


def threshold(real, ci_array):
    if (real <= ci_array[0]):
        return 1  # !!!
    elif (real >= ci_array[1]):
        return 1
    return 0


def uncorrected_test(val, alpha=.025):
    '''
    Perform an uncorrected voxel level non-paramatric test on data of 1 subject.
    :param val: np.array of dimension (num knockoffs, regions, paradigms). index 0 is the true value
    :param alpha: Two sided threshold, 2*alpha is the test ex: alpha .05 is 90% level
    :return: np.array (region, thresholded beta) 1 for right side reject, -1 for left side reject, 0 accept
    '''
    print("Performing uncorrected non-parametric test...")
    region = val.shape[1]
    n = val.shape[0]
    ci = [int(n * alpha), int(n * (1 - alpha))]  # confidence interval

    uncorrected_threshold = []
    # Loop over brain regions and at each region for each beta perform a hypothesis test
    for i in range(region):
        reg = val[:, i, :]
        real = reg[0, :]
        sort_reg = np.sort(reg, axis=0)
        ci_array = sort_reg[ci, :]
        ind = []
        # For each beta of the real subject: compare against confidence interval and accept or reject H0
        for b in range(len(real)):
            ind.append(threshold(real[b], ci_array[:, b]))
        uncorrected_threshold.append(ind)
    return np.array(uncorrected_threshold)


def corrected_test(val, alpha=.025):
    '''
    Perform a single threshold image wise non-paramatric test on data of 1 subject.
    :param val: np.array of dimension (num knockoffs, regions, knock_betas). index 0 is the true value
    :param alpha: Two sided threshold, 2*alpha is the test ex: alpha .05 is 90% level
    :return: np.array (region, thresholded beta) 1 for right side reject, -1 for left side reject, 0 accept
    '''
    print("Performing corrected non-parametric test...")
    paradigm = val.shape[2]
    regions = val.shape[1]
    ci = [int(regions * alpha), int(regions * (1 - alpha))] # confidence interval
    corrected_threshold = []

    # Loop over each paradigm and calculate maximal statistic for each beta over all brain regions.
    for i in range(paradigm):
        brain = val[:, :, i]
        real = brain[0, :]
        max_ = np.amax(brain, axis=0) # Takes the maximum per region
        min_ = np.amin(brain, axis=0) # Takes the minimum per region
        image_beta_max = np.sort(max_)
        image_beta_min = np.sort(min_)
        ci_array = [image_beta_min[ci[0]], image_beta_max[ci[1]]]

        ind = []
        # Compare beta values against maximal thresholded values and accept or reject null hypothesis.
        for reg in real:
            ind.append(threshold(reg, ci_array))
        corrected_threshold.append(ind)
    return np.array(corrected_threshold).T


def get_corrected_betas(corrected, betas):
    '''
    :param corrected: thresholded activations
    :param betas: beta values from the GLM
    :return: thresholded_betas: thresholded knock_betas after non-parametric tests
    '''
    thresholded_betas = corrected * betas
    return thresholded_betas


def nonparametric_test(task):
    """

    :param task:
    :return:
    """

    # TODO: do it for all subjects

    print(f"Starting Non-Parametric Tests for task {task}...")
    # Loading beta values for the knockoffs and for the true ones (from GLM)
    KNOCKOFF_PATH = f'./knockoff_af/betas/knockoff_test_betas_{task}.pickle'
    TRUE_PATH = f'./Data/beta_truth/beta_tfMRI_{task}_LR_Glasser360.mat'

    with open(KNOCKOFF_PATH, 'rb') as pickle_file:
        knock_betas = pickle.load(pickle_file)

    true_betas = scipy.io.loadmat(TRUE_PATH)['Beta'][:, :, 0]  # selecting subject 0
    # getting data into the following format: [region, timeseries]
    true_betas = np.swapaxes(true_betas, 0, 1)

    # with open(PATH2, 'rb') as infile:
    #     true_betas = pickle.load(infile)   # THIS DOESN'T WORK!!! it's like there is a blank space at the end


    # Calculating thresholded activations
    corrected_thresholded_activations = corrected_test(knock_betas)
    uncorrected_thresholded_activations = uncorrected_test(knock_betas)

    # Thresholding beta values using the previous thresholded activations
    thresholded_betas = get_corrected_betas(corrected_thresholded_activations, true_betas)

    # Saving files
    print(f"Saving thresholded activations and beta values for task {task}...")
    load.save_mat(thresholded_betas, 'Nonparametric_tests/activations_results', 'thresholded_betas', f'{task}_subj1')
    load.save_pickle(corrected_thresholded_activations, 'Nonparametric_tests/activations_results', 'thresholded_activations', "corr_test")
    load.save_pickle(uncorrected_thresholded_activations, 'Nonparametric_tests/activations_results', 'thresholded_activations', "uncorr_test")

    print("Done!")

#nonparametric_test('MOTOR')


''''
1) Right now the function just takes in one subject's stuff and outputs a pickle so have to 
create something (prob just a for loop) that will do this over all the subjects that we have. 
There will probably have to be a 4 -d array (Subjects, knockoffs, brain regions, paradigms)

2) how do we then take the thresholds and make it plotable so we can compare it with the GLM

3) can we numerically compare it with the GLM threshold. Can we use accuracy or something and say 
we classified x correctly and missed this many. Maybe something like what was our True Positive, False Postive, 
True Negative False negative

4) can we do the above visually?

5) need to figure out if there is a way to threshold the GLM with the pvalues over the image vs. at the voxel 
otherwise we aren't really comparing the same thing with our controlled test vs the GLM. We can ask Giulia this 
but can probably look something up as well

One should be easy, second should be easy as well, just multiple the beta array and the threshold array toghether 
and then send off to plot. But still have to have a function that does that and saves Mat files and such. 
3rd and 4th maybe a bit more more difficult
'''