from Knockoffs import knockoff_class
from Knockoffs.params import get_params
from input_output import load
import pickle

# k = knockoff_class.LowRankKnockOff('MOTOR', 0)
# k.load_fmri()
# k.fit()
# data = k.transform(save=True)


# k_so = knockoff_class.GaussianKnockOff('MOTOR', 0)
# k_so.load_fmri()
# k_so.pre_process(max_corr=.3)
# corr_g = k_so.fit()
# data_so = k_so.transform()
#
# sigma_hat, x = load.load_pickle('GaussianKO_tfMRI_tMOTOR_s0_c0.3.pickle')
# p = x.T.shape[1]
# n = x.T.shape[0]
# params = get_params(p, n, corr_g)

d = knockoff_class.DeepKnockOff('MOTOR', 0)
d.pre_process(max_corr=.3)
# d.fit()
d.load_machine()
data_deep = d.transform()