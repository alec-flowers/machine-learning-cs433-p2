from GLM import glm
import numpy as np
from input_output import load


tasks = ['MOTOR'] #, 'GAMBLING', 'RELATIONAL', 'SOCIAL', 'WM'] # 'EMOTION', 'LANGUAGE'] # TODO: see how to fix emotion and language
        # EMOTION is not working because fMRI has timempoints=176 and task paradigms has timepoints=186
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
    activations, betas, tvalues = glm.glm(fMRI, task_paradigms, hrf)

    # saving output for a specific task
    print(f"Saving activations and beta values for task {task}...")
    load.save_pickle(activations, 'act', 'act', task)
    load.save_pickle(betas, 'betas', 'betas', task)
    load.save_mat(betas, 'betas', 'betas', task)

    avg = np.mean(betas, axis=0)
    load.save_mat(avg, 'betas', 'avg_betas', task)
