import argparse
import pickle
from os.path import join
import numpy as np

from DeepKnockoffs import GaussianKnockoffs

from Knockoffs.params import DATA_PATH, KNOCKOFFS_PATH


def gaussian_knockoffs(task, subject, max_corr):
    file = f"tfMRI_{task}_s_{subject}_c_{max_corr}.pickle"
    path = join(DATA_PATH, file)
    try:
        with open(path, "rb") as f:
            SigmaHat, _ = pickle.load(f)
    except FileNotFoundError as e:
        FileNotFoundError(f"Need to pre-process data for the parameters tfMRI_{task}_s_{subject}_c_{max_corr} first!")
    # Initialize generator of second-order knockoffs
    second_order = GaussianKnockoffs(SigmaHat, mu=np.zeros((SigmaHat.shape[0])), method="sdp")
    # Measure pairwise second-order knockoff correlations
    corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)
    print('Average absolute pairwise correlation: %.3f.' % (np.mean(np.abs(corr_g))))
    file = f"gaussian_ko_{task}_s_{subject}_c_{max_corr}.pickle"
    path = join(DATA_PATH, KNOCKOFFS_PATH, file)
    with open(path, "wb") as f:
        pickle.dump(second_order, f)


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
    gaussian_knockoffs(args.task, args.subject, args.max_corr)
