import numpy as np
import scipy.cluster.hierarchy as spc

from DeepKnockoffs import GaussianKnockoffs
from deepknockoffs.examples import data
import fanok

from input_output import load


class KnockOff:
    hrf = load.load_hrf_function()

    def __init__(self, task, subject):
        assert task in ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM'], \
            'Task must be a value in - [EMOTION, GAMBLING, LANGUAGE, MOTOR, RELATIONAL, SOCIAL, WM]'
        assert isinstance(subject, int)
        assert (subject >= 0) & (subject <= 100)

        self.task = task
        self.subject = subject

        self.fmri = None
        self.paradigms = None
        self.generator = None

    def load_fmri(self):
        self.fmri = load.load_fmri(task=self.task)
        self.fmri = self.fmri[self.subject]

    def load_paradigms(self):
        self.paradigms = load.load_task_paradigms(task=self.task)
        self.paradigms = self.paradigms[self.subject, :]

    def pre_process(self):
        pass

    def check_data(self, x=None):
        if not x:
            if self.fmri is None:
                self.load_fmri()
            x = self.fmri.T
        return x

    def fit(self, x=None):
        pass

    def transform(self, x=None, iters=100):
        pass

    def diagnostics(self):
        pass


class LowRankKnockOff(KnockOff):
    def __init__(self, task, subject):
        super().__init__(task, subject)

    def fit(self, x=None, rank=50):
        x = super().check_data(x)
        factor_model = fanok.RandomizedLowRankFactorModel(rank=rank)
        self.generator = fanok.LowRankGaussianKnockoffs(factor_model)
        self.generator.fit(X=x)

    def transform(self, x=None, iters=100):
        x = super().check_data(x)
        all_knockoff = np.zeros((iters, x.shape[0], x.shape[1]))
        for i in range(iters):
            knock = self.generator.transform(X=x)
            all_knockoff[i, :, :] = knock

        all_knockoff = np.concatenate((np.expand_dims(x, axis=0), all_knockoff), axis=0)
        return all_knockoff

class GaussianKnockOff(KnockOff):
    def __init__(self, task, subject, max_corr):
        super().__init__(task, subject)
        self.max_corr = max_corr

    def fit(self, sigma_hat):
        # Initialize generator of second-order knockoffs
        self.generator = GaussianKnockoffs(sigma_hat, mu=np.zeros((sigma_hat.shape[0])), method="sdp")
        # Measure pairwise second-order knockoff correlations
        corr_g = (np.diag(sigma_hat) - np.diag(self.generator.Ds)) / np.diag(sigma_hat)
        print('Average absolute pairwise correlation: %.3f.' % (np.mean(np.abs(corr_g))))
        return corr_g

    def transform(self):