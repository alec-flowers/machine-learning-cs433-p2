import argparse
import pickle
from os.path import join
import numpy as np
from DeepKnockoffs import KnockoffMachine
from deepknockoffs.examples.diagnostics import ScatterCovariance, COV, KNN, Energy, MMD, compute_diagnostics
import matplotlib.pyplot as plt
import torch

# constants
DATA_PATH = "../Data"
KNOCKOFFS_PATH = "Knockoffs"


def diagnostics(task, subject, max_corr):
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
    # Set the parameters for training deep knockoffs
    pars = dict()
    # Number of epochs
    pars['epochs'] = 100
    # Number of iterations over the full data per epoch
    pars['epoch_length'] = 100
    # Data type, either "continuous" or "binary"
    pars['family'] = "continuous"
    # Dimensions of the data
    pars['p'] = p
    # Size of the test set
    pars['test_size'] = 0
    # Batch size
    pars['batch_size'] = int(0.5 * n)
    # Learning rate
    pars['lr'] = 0.01
    # When to decrease learning rate (unused when equal to number of epochs)
    pars['lr_milestones'] = [pars['epochs']]
    # Width of the network (number of layers is fixed to 6)
    pars['dim_h'] = int(10 * p)
    # Penalty for the MMD distance
    pars['GAMMA'] = 1.0
    # Penalty encouraging second-order knockoffs
    pars['LAMBDA'] = 0.1
    # Decorrelation penalty hyperparameter
    pars['DELTA'] = 0.1
    # Target pairwise correlations between variables and knockoffs
    pars['target_corr'] = corr_g
    # Kernel widths for the MMD measure (uniform weights)
    pars['alphas'] = [1., 2., 4., 8., 16., 32., 64., 128.]

    # Where the machine is stored
    file = f"deep_ko_{task}_s_{subject}_c_{max_corr}"
    checkpoint_name = join(DATA_PATH, KNOCKOFFS_PATH, file)
    # Initialize the machine
    machine = KnockoffMachine(pars)
    # Load the machine
    machine.load(checkpoint_name)

    # plot diagnostics for second order knockoffs
    Xk_train_g = second_order.generate(X_train)
    # # Divide data into batches
    # n1 = int(X_train.shape[0] / 2)
    # X1, Xk1 = X_train[:n1], Xk_train_g[:n1]
    # X2, Xk2 = X_train[n1:], Xk_train_g[n1:]
    # Z_ref = torch.cat((torch.from_numpy(X1), torch.from_numpy(Xk1)), 1)
    # Z_ref2 = torch.cat((torch.from_numpy(X2), torch.from_numpy(Xk2)), 1)
    ScatterCovariance(X_train, Xk_train_g)
    plt.savefig("scatter_cov_gaussian_ko.pdf", format="pdf")
    # results = MMD(Z_ref, Z_ref2, [1., 2., 4., 8., 16., 32., 64., 128.])
    print("Size of the second-order knockoff dataset: %d x %d." % (Xk_train_g.shape))
    # plot diagnostics for deep knockoffs
    Xk_train_g = machine.generate(X_train)
    ScatterCovariance(X_train, Xk_train_g)
    plt.savefig("scatter_cov_deep_ko.pdf", format="pdf")
    print("Size of the deep knockoff dataset: %d x %d." % (Xk_train_g.shape))


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
    diagnostics(args.task, args.subject, args.max_corr)
