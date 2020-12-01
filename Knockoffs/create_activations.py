import pickle
from os.path import join
import numpy as np

from DeepKnockoffs import KnockoffMachine
from Knockoffs.params import DATA_PATH, KNOCKOFFS_PATH, get_params

from Knockoffs.generate_knockoffs import do_generate, populate_with_representatives
from input_output import load
from GLM import glm

# init vars
task = 'MOTOR'
subject = 0
max_corr = 0.2
ko_type = "second_order"

# reading input files
file = f"tfMRI_{task}_s_{subject}_c_{max_corr}.pickle"
path = join(DATA_PATH, file)
with open(path, "rb") as f:
    SigmaHat, X_train = pickle.load(f)
X_train = X_train.T
if ko_type == "second_order" or "machine":
    file = f"gaussian_ko_{task}_s_{subject}_c_{max_corr}.pickle"
    path = join(DATA_PATH, KNOCKOFFS_PATH, file)
    with open(path, "rb") as f:
        knockoff_generator = pickle.load(f)
    if ko_type == "machine":
        # Measure pairwise second-order knockoff correlations
        corr_g = (np.diag(SigmaHat) - np.diag(knockoff_generator.Ds)) / np.diag(SigmaHat)
        p = X_train.shape[1]
        n = X_train.shape[0]
        pars = get_params(p, n, corr_g)
        # Where the machine is stored
        file = f"deep_ko_{task}_s_{subject}_c_{max_corr}"
        checkpoint_name = join(DATA_PATH, KNOCKOFFS_PATH, file)
        # Initialize the machine
        knockoff_generator = KnockoffMachine(pars)
        # Load the machine
        knockoff_generator.load(checkpoint_name)
file = f"mapping_{task}_s_{subject}_c_{max_corr}.pickle"
path = join(DATA_PATH, file)
with open(path, "rb") as f:
    groups, representatives = pickle.load(f)

# generate 100 knockoffs
num = 100
for i in range(num):
    Xk = do_generate(knockoff_generator.generate, X_train, task, subject, max_corr)
    Xk = populate_with_representatives(groups, representatives, Xk)
    if i == 0:
        knock_feat = np.zeros((num, Xk.shape[0], Xk.shape[1]))
    knock_feat[i] = Xk

# preparing X with knockoffs
fMRI = load.load_fmri(task=task)
X = fMRI[subject]
knock_feat = np.concatenate((np.expand_dims(X, axis=0), knock_feat), axis=0)
# doing the glm
task_paradigms = np.expand_dims(load.load_task_paradigms(task)[subject], axis=0)
task_paradigms = np.repeat(task_paradigms, num + 1, axis=0)
hrf = load.load_hrf_function()
activations, betas, tvalues = glm.glm(knock_feat, task_paradigms, hrf)
# saving output
print(f"Saving activations and beta values for task {task}...")
load.save_pickle(activations, 'GLM/activations', 'activation_knockoffs', task)
load.save_pickle(betas, 'GLM/betas', 'betas_knockoffs', task)
load.save_mat(betas, 'GLM/betas', 'betas_knockoffs', task)
