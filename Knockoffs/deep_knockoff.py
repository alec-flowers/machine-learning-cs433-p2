import argparse
import pickle
from os.path import join
import numpy as np
from DeepKnockoffs import KnockoffMachine
from Knockoffs.params import get_params

from Knockoffs.params import DATA_PATH, KNOCKOFFS_PATH


def deep_knockoff(task, subject, max_corr):
    # reading input files
    file = f"tfMRI_{task}_s_{subject}_c_{max_corr}.pickle"
    path = join(DATA_PATH, file)
    try:
        with open(path, "rb") as f:
            SigmaHat, X_train = pickle.load(f)
    except FileNotFoundError as e:
        FileNotFoundError(f"Need to pre-process data for the parameters tfMRI_{task}_s_{subject}_c_{max_corr} first!")
    file = f"gaussian_ko_{task}_s_{subject}_c_{max_corr}.pickle"
    path = join(DATA_PATH, KNOCKOFFS_PATH, file)
    try:
        with open(path, "rb") as f:
            second_order = pickle.load(f)
    except FileNotFoundError as e:
        FileNotFoundError(
            f"Need to create gaussian knockoffs for the parameters tfMRI_{task}_s_{subject}_c_{max_corr} first!")

    # Measure pairwise second-order knockoff correlations
    corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)
    # training the neural net
    X_train = X_train.T
    p = X_train.shape[1]
    n = X_train.shape[0]
    pars = get_params(p, n, corr_g)
    # Where to store the machine
    file = f"deep_ko_{task}_s_{subject}_c_{max_corr}"
    checkpoint_name = join(DATA_PATH, KNOCKOFFS_PATH, file)
    # Where to print progress information
    logs_name = join(DATA_PATH, KNOCKOFFS_PATH, file + "_progress.txt")
    # Initialize the machine
    machine = KnockoffMachine(pars, checkpoint_name=checkpoint_name, logs_name=logs_name)
    # Train the machine
    print("Fitting the knockoff machine...")
    machine.train(X_train)


def parse_args():
    parser = argparse.ArgumentParser(
        description="For which pre-processed data to build the Gaussian Knockoff Generator")
    parser.add_argument('-t', '--task', type=str, help='Which task set to load.', required=True,
                        choices=['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM'])
    parser.add_argument('-s', '--subject', type=int, help='Which to pre-process.', required=True)
    parser.add_argument('-c', '--max_corr', type=float, help="Maximum allowed correlation in clustering", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    deep_knockoff(args.task, args.subject, args.max_corr)
