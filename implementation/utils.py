import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import os.path
import numpy as np
import scipy.cluster.hierarchy as spc

REPO_ROOT = pathlib.Path(__file__).absolute().parents[1].absolute().resolve()
assert (REPO_ROOT.exists())
DATA_DIR = (REPO_ROOT / "data").absolute().resolve()
assert (DATA_DIR.exists())
INPUT_DIR = (DATA_DIR / "input").absolute().resolve()
assert (INPUT_DIR.exists())
OUTPUT_DIR = (DATA_DIR / "output").absolute().resolve()
assert (OUTPUT_DIR.exists())
KNOCK_DIR = (OUTPUT_DIR / "knockoffs").absolute().resolve()
assert (KNOCK_DIR.exists())
IMG_DIR = (OUTPUT_DIR / "img").absolute().resolve()
assert (IMG_DIR.exists())
BETA_DIR = (OUTPUT_DIR / "beta").absolute().resolve()
assert (BETA_DIR.exists())


def plot_goodness_of_fit(results, metric, title, name, swap_equals_self=False, save_img=True):
    if not swap_equals_self:
        file = f"{name}_box_{metric}.pdf"
    else:
        file = f"{name}_box_corr.pdf"
    fig, ax = plt.subplots(figsize=(12, 6))
    do_plot(results, metric, swap_equals_self, ax)
    ax.set_title(title)
    file_path = os.path.join(IMG_DIR, file)
    if save_img:
        plt.savefig(file_path, format="pdf")
    return fig, ax


def do_plot(results, metric, swap_equals_self, ax):
    if not swap_equals_self:
        data = results[(results.Metric == metric) & (results.Swap != "self")]
    else:
        data = results[(results.Metric == metric) & (results.Swap == "self")]
    sns.boxplot(x="Swap", y="Value", hue="Method", data=data, ax=ax)
    return ax


def do_pre_process(X, max_corr):
    from deepknockoffs.examples import data
    # calc SigmaHat
    SigmaHat = np.cov(X)
    Corr = data.cov2cor(SigmaHat)
    # Compute distance between variables based on their pairwise absolute correlations
    pdist = spc.distance.squareform(1 - np.abs(Corr))
    # Apply average-linkage hierarchical clustering
    linkage = spc.linkage(pdist, method='average')
    corr_max = max_corr
    d_max = 1 - corr_max
    # Cut the dendrogram and define the groups of variables
    groups = spc.cut_tree(linkage, height=d_max).flatten()
    print("Divided " + str(len(groups)) + " variables into " + str(np.max(groups) + 1) + " groups.")
    linkage = spc.linkage(pdist, method='average')
    print("Divided " + str(len(groups)) + " variables into " + str(np.max(groups) + 1) + " groups.")
    # Plot group sizes
    _, counts = np.unique(groups, return_counts=True)
    print("Size of largest groups: " + str(np.max(counts)))
    print("Mean groups size: " + str(np.mean(counts)))
    # Pick one representative for each cluster
    representatives = np.array([np.where(groups == g)[0][0] for g in
                                np.arange(np.max(groups) + 1)])  # + 1 due to np.arange(), bug in original code
    # Sigma Hat matrix for group representatives
    SigmaHat_repr = SigmaHat[representatives, :][:, representatives]
    # Correlations for group representatives
    Corr_repr = data.cov2cor(SigmaHat_repr)
    # fMRI representatives
    X_repr = X[representatives]
    print(f"Eigenvalue for Sigma Hat, Min: {np.min(np.linalg.eigh(SigmaHat)[0])}")
    print(f"Eigenvalue for Sigma Hat Representatives, Min: {np.min(np.linalg.eigh(SigmaHat_repr)[0])}")
    print(f"Original for Correlations, Max: {np.max(np.abs(Corr - np.eye(Corr.shape[0])))}")
    print(f"Representatives for Correlations, Max: {np.max(np.abs(Corr_repr - np.eye(Corr_repr.shape[0])))}")
    return SigmaHat_repr, X_repr, groups, representatives
