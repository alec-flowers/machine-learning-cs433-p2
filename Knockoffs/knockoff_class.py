import numpy as np
import scipy.cluster.hierarchy as spc
import pickle
from os.path import join
import abc

from DeepKnockoffs import GaussianKnockoffs, KnockoffMachine
from deepknockoffs.examples import data
from Knockoffs.params import get_params
import fanok
from input_output import load
from .utils import REPO_ROOT, DATA_DIR, KNOCK_DIR

#TODO - DeepKO transform, diagnostics incorporate, GLM and threshold maybe, possibly refactor to not do any data stuff in class itself.

class KnockOff(abc.ABC):

    def __init__(self, task=None, subject=None):
        assert task in ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM'], \
            'Task must be a value in - [EMOTION, GAMBLING, LANGUAGE, MOTOR, RELATIONAL, SOCIAL, WM]'
        assert isinstance(subject, int)
        assert (subject >= 0) & (subject <= 100)

        self.task = task
        self.subject = subject

        self.NAME = None
        self.x_train = None
        self.paradigms = None
        self.generator = None
        self.file = None

    def load_fmri(self):
        self.x_train = load.load_fmri(task=self.task)
        self.x_train = self.x_train[self.subject]

    def load_paradigms(self):
        self.paradigms = load.load_task_paradigms(task=self.task)
        self.paradigms = self.paradigms[self.subject, :]

    @staticmethod
    def save_pickle(file, to_pickle):
        path = join(DATA_DIR, file)
        with open(path, "wb") as f:
            pickle.dump(to_pickle, f)

    def check_data(self, x=None, transpose=False):
        if x is not None:
            self.x_train = x
        if self.x_train is None:
            ValueError('x cannot be None. Provide data or use load_fmri()')

        if transpose:
            x = self.x_train.T
        else:
            x = self.x_train
        return x

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def generate(self, x):
        pass

    def transform(self, x=None, iters=100, save=False):
        x = self.check_data(x, transpose=True)
        all_knockoff = np.zeros((iters, x.shape[0], x.shape[1]))
        for i in range(iters):
            knock = self.generate(x)
            all_knockoff[i, :, :] = knock

        all_knockoff = np.concatenate((np.expand_dims(x, axis=0), all_knockoff), axis=0)
        if save:
            self.save_pickle(self.NAME+'_KO_'+self.file, all_knockoff)
        return all_knockoff

    def diagnostics(self):
        pass


class LowRankKnockOff(KnockOff):

    def __init__(self, task, subject):
        super().__init__(task, subject)
        self.file = f"t{task}_s{subject}.pickle"
        self.NAME = 'LowRankKO'

    def fit(self, x=None, rank=50):
        x = self.check_data(x, transpose=True)
        factor_model = fanok.RandomizedLowRankFactorModel(rank=rank)
        self.generator = fanok.LowRankGaussianKnockoffs(factor_model)
        self.generator.fit(X=x)

    def generate(self, x):
        return self.generator.transform(X=x)

    def transform(self, x=None, iters=100, save=False):
        all_knockoff = super().transform(x=x, iters=iters, save=save)
        return all_knockoff


def do_pre_process(X, max_corr):
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


class GaussianKnockOff(KnockOff):
    def __init__(self, task, subject):
        super().__init__(task, subject)
        self.NAME = 'GaussianKO'
        self.max_corr = None
        self.sigma_hat = None
        self.x_train = None
        self.corr_g = None
        self.groups = None

    def pre_process(self, max_corr, x=None, save=False):
        self.max_corr = max_corr
        self.file = f"t{self.task}_s{self.subject}_c{self.max_corr}.pickle"

        x = self.check_data(x)
        self.sigma_hat, self.x_train, self.groups, representatives = do_pre_process(x, self.max_corr)

        if save:
            self.save_pickle(self.NAME +'_tfMRI_' + self.file, (self.sigma_hat, self.x_train))
            self.save_pickle(self.NAME+'_mapping_'+self.file, (self.groups, representatives))

    def fit(self, sigma_hat=None, save=False):
        if sigma_hat is not None:
            self.sigma_hat = sigma_hat
        if self.sigma_hat is None:
            raise ValueError("Sigma Hat cannot be None.")

        # Initialize generator of second-order knockoffs
        self.generator = GaussianKnockoffs(self.sigma_hat, mu=np.zeros((self.sigma_hat.shape[0])), method="sdp")
        # Measure pairwise second-order knockoff correlations
        self.corr_g = (np.diag(self.sigma_hat) - np.diag(self.generator.Ds)) / np.diag(self.sigma_hat)
        print('Average absolute pairwise correlation: %.3f.' % (np.mean(np.abs(self.corr_g))))

        if save:
            self.save_pickle(self.NAME+'_SecOrd_'+self.file, self.generator)
        return self.corr_g

    def generate(self, x):
        return self.generator.generate(x)

    def transform(self, x=None, iters=100, save=False):
        all_knockoff = super().transform(x=x, save=True)
        return all_knockoff


class DeepKnockOff(KnockOff):
    def __init__(self, task, subject, params=None):
        super().__init__(task, subject)
        self.params = params

        self.NAME = 'DeepKO'
        self.file = f"{self.NAME}_t{task}_s{subject}"

    def pre_process(self, max_corr):
        assert self.params is None, 'Params already exists, this would override params.'

        gauss = GaussianKnockOff(self.task, self.subject)
        gauss.load_fmri()
        gauss.pre_process(max_corr=max_corr)
        self.x_train = gauss.x_train
        corr_g = gauss.fit()

        p = self.x_train.T.shape[1]
        n = self.x_train.T.shape[0]
        self.params = get_params(p, n, corr_g)

    def load_machine(self):
        assert self.params is not None, ValueError('Params cannot be None. Please pass in params or run pre-process()')
        checkpoint_name = join(KNOCK_DIR, self.file)
        self.generator = KnockoffMachine(self.params)
        self.generator.load(checkpoint_name)

    def fit(self, x=None):
        if self.generator is not None:
            raise ValueError('Trained generator already exists')
        x = self.check_data(x, transpose=True)

        checkpoint_name = join(KNOCK_DIR, self.file)
        # Where to print progress information
        logs_name = join(KNOCK_DIR, self.file + "_progress.txt")
        # Initialize the machine
        self.generator = KnockoffMachine(self.params, checkpoint_name=checkpoint_name, logs_name=logs_name)
        # Train the machine
        print("Fitting the knockoff machine...")
        self.generator.train(x)
        return self.generator

    def generate(self, x):
        return self.generator.generate(x)

    def transform(self, x=None, iters=100, save=False):
        if self.generator is None:
            raise ValueError("Generator cannot be None. Use load_machine() or fit() to train the generator")
        all_knockoff = super().transform(x=x, save=True)
        return all_knockoff

