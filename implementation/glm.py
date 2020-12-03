import implementation.load as load
import numpy as np
import statsmodels.api as sm


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

    # Reshaping so that fMRI and task_paradigms shapes match by keeping the shorter length
    if fMRI.shape[2] != task_paradigms.shape[1]:
        fMRI = fMRI[:, :, :min(fMRI.shape[2], task_paradigms.shape[1])]
        task_paradigms = task_paradigms[:, :min(fMRI.shape[2], task_paradigms.shape[1])]

    assert fMRI.shape[2] == task_paradigms.shape[1], \
        f"fMRI and task_paradigms shapes do not match: {fMRI.shape[1]} and {task_paradigms.shape}"

    # do one hot encoding
    task_paradigms_one_hot = load.separate_conditions(task_paradigms)

    # do the convolution
    task_paradigms_conv = load.do_convolution(task_paradigms_one_hot, hrf)

    # fit the glm for every subject and region
    print(f"Fitting GLM for {fMRI.shape[0]} subjects and {fMRI.shape[1]} regions...")
    p_value = 0.05
    activations = np.zeros((task_paradigms_one_hot.shape[0], fMRI.shape[1], task_paradigms_one_hot.shape[1] - 1))
    betas = np.zeros((task_paradigms_one_hot.shape[0], fMRI.shape[1], task_paradigms_one_hot.shape[1] - 1))
    tvalues = np.zeros((task_paradigms_one_hot.shape[0], fMRI.shape[1], task_paradigms_one_hot.shape[1] - 1))

    for subject in range(fMRI.shape[0]):
        for region in range(fMRI.shape[1]):
            # dropping the first one hot encoding with 1: to circumvent dummy variable problem
            # X = sm.add_constant(np.swapaxes(task_paradigms_conv[subject, 1:, :], 0, 1))
            X = np.swapaxes(task_paradigms_conv[subject, 1:, :], 0, 1)
            y = fMRI[subject, region, :]
            mod = sm.OLS(y, X)
            res = mod.fit()
            p_values = res.pvalues
            coef = res.params
            tval = res.tvalues
            # prints RuntimeError when y==0, i.e. all coefficients of the OLS are zero
            activations[subject, region, :] = p_values < p_value
            betas[subject, region, :] = coef
            tvalues[subject, region, :] = tval
    ### TODO: we need to only select the betas which are active, right?! rn we are not taking
    ### into account if that beta is significant or not, because we only take into accunt the
    ### pvalue for the activations
    active_betas = activations * betas

    print("Done!")

    return activations, betas, tvalues, active_betas


if __name__ == "__main__":
    tasks = [
        'MOTOR']  # , 'GAMBLING', 'RELATIONAL', 'SOCIAL', 'WM'] # 'EMOTION', 'LANGUAGE'] # TODO: see how to fix emotion and language
    # EMOTION is not working because fMRI has timepoints=176 and task paradigms has timepoints=186
    # LANGUAGE is not working because each subject of the task paradigm has different length of timeseries
    hrf = load.load_hrf_function()

    for task in tasks:
        # loading data for a specific task
        print(f'============================== \n {task} \n==============================')
        print(f"Loading data for task {task}...")
        fMRI = load.load_fmri(task)
        task_paradigms = load.load_task_paradigms(task)

        # computing glm for a specific task
        print(f'Computing GLM for task {task}...')
        activations, betas, tvalues, active_betas = glm(fMRI, task_paradigms, hrf)

        # saving output for a specific task
        print(f"Saving activations and beta values for task {task}...")
        load.save_pickle(activations, 'activations', 'activation', task)
        load.save_pickle(betas, 'betas', 'betas', task)
        load.save_mat(betas, 'betas', 'betas', task)

        avg = np.mean(betas, axis=0)
        load.save_mat(avg, 'betas', 'avg_betas', task)
