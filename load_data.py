import os
import numpy as np
import scipy.io

def load_hrf(task = 'MOTOR', filepath = './Data/'):
    'Load the hemodynamic response function which is a 3-d array.'
    assert task in ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM'],\
        'Task must be a value in - [EMOTION, GAMBLING, LANGUAGE, MOTOR, RELATIONAL, SOCIAL, WM]'

    data = scipy.io.loadmat(filepath+'X_tfMRI_'+task+'_LR_Glasser360.mat')
    data = data['X']

    print(f'Loaded Data - Shape: {data.shape}')
    return data

def load_task_paradigms(task = 'MOTOR', directory = './Data/TaskParadigms'):
    'Load all the task paradigms.'
    assert task in ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM'],\
        'Task must be a value in - [EMOTION, GAMBLING, LANGUAGE, MOTOR, RELATIONAL, SOCIAL, WM]'

    FILE = task+'_LR.mat'
    OPEN = directory + '/'

    regressor = {}
    for filename in os.listdir(directory):
        if filename.endswith(FILE):
            task = scipy.io.loadmat(OPEN+filename)
            regressor[filename.split('_')[0]] = task['Regressor']

    print(f'Loaded Task Paradigms - Length: {len(regressor)}')
    return regressor

if __name__ == '__main__':
    load_hrf()
    load_task_paradigms()
