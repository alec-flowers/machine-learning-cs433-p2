import argparse
import pickle
from os.path import join
from collections import defaultdict
import numpy as np
import pandas as pd
from DeepKnockoffs import KnockoffMachine
from deepknockoffs.examples.diagnostics import ScatterCovariance, compute_diagnostics
import matplotlib.pyplot as plt  #
import seaborn as sns
import torch
from Knockoffs.params import get_params

from Knockoffs.params import DATA_PATH, KNOCKOFFS_PATH, ALPHAS


def generate_knockoff(task, subject, max_corr, ko_type):
    # reading input files
    file = f"tfMRI_{task}_s_{subject}_c_{max_corr}.pickle"
    path = join(DATA_PATH, file)
    with open(path, "rb") as f:
        SigmaHat, X_train = pickle.load(f)
    file = f"gaussian_ko_{task}_s_{subject}_c_{max_corr}.pickle"
    path = join(DATA_PATH, KNOCKOFFS_PATH, file)
    with open(path, "rb") as f:
        second_order = pickle.load(f)

    # Measure pairwise second-order knockoff correlations
    corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)
    # training the neural net
    X_train = X_train.T
    p = X_train.shape[1]
    n = X_train.shape[0]
    pars = get_params(p, n, corr_g)

    # Where the machine is stored
    file = f"deep_ko_{task}_s_{subject}_c_{max_corr}"
    checkpoint_name = join(DATA_PATH, KNOCKOFFS_PATH, file)
    # Initialize the machine
    machine = KnockoffMachine(pars)
    # Load the machine
    machine.load(checkpoint_name)

    if ko_type == "second_order":
        Xk_train_g = second_order.generate(X_train)
    elif ko_type == "deep":
        Xk_train_g = machine.generate(X_train)

    file = f"mapping_{task}_s_{subject}_c_{max_corr}.pickle"
    path = join(DATA_PATH, file)
    with open(path, "rb") as f:
        groups, representatives = pickle.load(f)
    Xk_train_g = Xk_train_g.T
    Xk_full = np.zeros((groups.shape[0], Xk_train_g.shape[1]))
    for region, my_group in enumerate(groups):
        Xk_full[region] = Xk_train_g[my_group, :]
    return Xk_full


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
    generate_knockoff(args.task, args.subject, args.max_corr, "second_order")
