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


def plot_goodness_of_fit(results, metric, title, swap_equals_self=False):
    if not swap_equals_self:
        data = results[(results.Metric == metric) & (results.Swap != "self")]
        file = f"box_{metric}.pdf"
    else:
        data = results[(results.Metric == metric) & (results.Swap == "self")]
        file = f"box_corr.pdf"
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x="Swap", y="Value", hue="Method", data=data)
    plt.title(title)
    plt.savefig(file, format="pdf")


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
    pars = get_params(p, n, corr_g)

    # Where the machine is stored
    file = f"deep_ko_{task}_s_{subject}_c_{max_corr}"
    checkpoint_name = join(DATA_PATH, KNOCKOFFS_PATH, file)
    # Initialize the machine
    machine = KnockoffMachine(pars)
    # Load the machine
    machine.load(checkpoint_name)

    X_train_tensor = torch.from_numpy(X_train).double()
    do_diagnostics(X_train, X_train_tensor, machine.generate, second_order=second_order)


def do_diagnostics(X_train, X_train_tensor, generate_f, second_order=None):
    """
    :param X_train:
    :param X_train_tensor: Make sure X_train_tensor is torch.from_numpy(X_train).double()
    :param generate_f:
    :param second_order:
    :return:
    """
    results = pd.DataFrame(columns=['Method', 'Metric', 'Swap', 'Value', 'Sample'])
    alphas = ALPHAS
    n_exams = 100
    for exam in range(n_exams):
        # diagnostics for deep knockoffs
        machine_name = "machine"
        Xk_train_g = generate_f(X_train)
        Xk_train_g_tensor = torch.from_numpy(Xk_train_g).double()
        new_res = compute_diagnostics(X_train_tensor, Xk_train_g_tensor, alphas)
        new_res["Method"] = machine_name
        new_res["Sample"] = exam
        results = results.append(new_res)
        if exam == 0:
            ScatterCovariance(X_train, Xk_train_g)
            plt.title("Covariance Scatter Plot Deep Knockoffs")
            plt.savefig("scatter_cov_deep_ko.pdf", format="pdf")

        if second_order is not None:
            # diagnostics for second order knockoffs
            machine_name = "second"
            Xk_train_g = second_order.generate(X_train)
            Xk_train_g_tensor = torch.from_numpy(Xk_train_g).double()
            new_res = compute_diagnostics(X_train_tensor, Xk_train_g_tensor, alphas)
            new_res["Method"] = machine_name
            new_res["Sample"] = exam
            results = results.append(new_res)
            if exam == 0:
                ScatterCovariance(X_train, Xk_train_g)
                plt.title("Covariance Scatter Plot Gaussian Knockoffs")
                plt.savefig("scatter_cov_gaussian_ko.pdf", format="pdf")

    print(results.groupby(['Method', 'Metric', 'Swap']).describe())
    for metric, title, swap_equals_self in zip(["Covariance", "KNN", "MMD", "Energy", "Covariance"],
                                               ["Covariance Goodness-of-Fit", "KNN Goodness-of-Fit",
                                                "MMD Goodness-of-Fit",
                                                "Energy Goodness-of-Fit",
                                                "Absolute Average Pairwise Correlations between Variables and Knockoffs"],
                                               [False, False, False, False, True]):
        plot_goodness_of_fit(results, metric, title, swap_equals_self)


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
