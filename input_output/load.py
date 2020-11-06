import os
from collections import OrderedDict
import scipy.io
import numpy as np


def load_hrf(task='MOTOR', filepath='../Data/'):
    """Load the hemodynamic response function which is a 3-d array."""
    assert task in ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM'], \
        'Task must be a value in - [EMOTION, GAMBLING, LANGUAGE, MOTOR, RELATIONAL, SOCIAL, WM]'

    data = scipy.io.loadmat(filepath + 'X_tfMRI_' + task + '_LR_Glasser360.mat')
    data = data['X']
    # getting data into the following format: [subject, region, timeseries]
    data = np.swapaxes(data, 1, 2)
    data = np.swapaxes(data, 0, 1)

    print(f'Loaded Data - Shape: {data.shape}')
    return data


def load_task_paradigms(task='MOTOR', directory='../Data/TaskParadigms'):
    """Load all the task paradigms."""
    assert task in ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM'], \
        'Task must be a value in - [EMOTION, GAMBLING, LANGUAGE, MOTOR, RELATIONAL, SOCIAL, WM]'

    FILE = task + '_LR.mat'
    OPEN = directory + '/'

    regressor = {}
    for filename in os.listdir(directory):
        if filename.endswith(FILE):
            task = scipy.io.loadmat(OPEN + filename)
            regressor[filename.split('_')[0]] = task['Regressor']

    regressor = OrderedDict(sorted(regressor.items()))
    regressor = np.array(list(regressor.values())).squeeze()

    print(f'Loaded Task Paradigms - Length: {len(regressor)}')
    return regressor


def load_hrf_function():
    """
    Load the hrf function given by Giulia.
    :return:
    """
    hrf = scipy.io.loadmat('../Data/hrf.mat')['hrf'].squeeze()
    pad = np.zeros(10, dtype=hrf.dtype)
    hrf_padded = np.concatenate((pad, hrf, pad))

    print(f"Loaded HRF and padded with 10 0's- Length: {len(hrf_padded)}")
    return hrf_padded


def separate_conditions(task_paradigms):
    """
    Separates the conditions given in task_paradigms
    :param task_paradigms: nd_array(n_subjects, n_timeseries)
    :return: nd_array(n_subjects, n_conditions, n_timeseries)
    """
    print("Separating conditions...")
    n_conditions = np.max(task_paradigms) + 1
    task_paradigms_one_hot = np.zeros((task_paradigms.shape[0], n_conditions, task_paradigms.shape[1]))
    for subject in range(task_paradigms.shape[0]):
        s = task_paradigms[subject].squeeze()
        b = np.zeros((s.size, s.max() + 1))
        b[np.arange(s.size), s] = 1
        task_paradigms_one_hot[subject] = b.transpose()
    print("Done!")
    return task_paradigms_one_hot


def do_convolution(task_paradigms_one_hot, hrf):
    """
    Performs the convolution with a HRF and the seperate task paradigms.
    :param task_paradigms_one_hot: nd_array(n_subjects, n_conditions, n_timeseries)
    :param hrf: the HRF to perform the conv. with
    :return: nd_array(n_subjects, n_conditions, n_timeseries)
    """
    print("Convolving...")
    task_paradigms_conv = np.zeros(task_paradigms_one_hot.shape)
    for subject in range(task_paradigms_one_hot.shape[0]):
        for condition in range(task_paradigms_one_hot.shape[1]):
            convolution = np.convolve(task_paradigms_one_hot[subject, condition, :], hrf, "full")
            # convolution contains len(hrf) + len(task_paradigms_one_hot[subject, condition, :]) + 1 elements, needs to
            # reduce size to len(ask_paradigms_one_hot[subject, condition, :]) as given in Giulias code fragment
            task_paradigms_conv[subject, condition, :] = convolution[:task_paradigms_one_hot.shape[2]]
    print("Done!")
    return task_paradigms_conv
