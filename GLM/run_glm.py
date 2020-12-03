from GLM import glm
import numpy as np
from input_output import load


tasks = ['MOTOR', 'GAMBLING', 'RELATIONAL', 'SOCIAL', 'WM', 'EMOTION', 'LANGUAGE']
        # EMOTION fMRI has timepoints=176 and task paradigms has timepoints=186
        # LANGUAGE each subject of the task paradigm has different length of timeseries
hrf = load.load_hrf_function()

for task in tasks:
    # loading data for a specific task
    print(f'============================== \n {task} \n==============================')
    print(f"Loading data for task {task}...")
    fMRI = load.load_fmri(task)
    task_paradigms = load.load_task_paradigms(task)

    # computing glm for a specific task
    print(f'Computing GLM for task {task}...')
    activations, betas, tvalues, active_betas = glm.glm(fMRI, task_paradigms, hrf)  #!!!


    # saving output for a specific task
    print(f"Saving activations and beta values for task {task}...")
    load.save_pickle(activations, 'GLM/activations', 'activation', task)
    load.save_pickle(betas, 'GLM/betas', 'betas', task)
    load.save_mat(betas, 'GLM/betas', 'betas', task)
    load.save_mat(active_betas, 'GLM/betas', 'active_betas', task)

