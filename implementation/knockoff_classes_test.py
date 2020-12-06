import knockoff_class
from implementation.load import load_pickle
from implementation.utils import KNOCK_DIR

# Example of running LowRankKnockOff

# k = knockoff_class.LowRankKnockOff('MOTOR', 0)
# k.load_fmri()
# k.fit(rank=120)
# k.diagnostics()
# data = k.transform()
# ko_betas = k.statistic(data, save=True)
# uncorrected_betas, corrected_betas = k.threshold(ko_betas, save=True)
#
# # Example of running GaussianKnocckOff
#
# k_so = knockoff_class.GaussianKnockOff('MOTOR', 0)
# k_so.load_fmri()
# k_so.pre_process(max_corr=.3)
# corr_g = k_so.fit()
# k_so.diagnostics()
# data_so = k_so.transform()
# ko_betas = k_so.statistic(data_so, save=True)
# uncorrected_betas, corrected_betas = k_so.threshold(ko_betas, save=True)

#
# # Example of running DeepKnockOff
#
_, x_train = load_pickle(KNOCK_DIR, 'GaussianKO_tfMRI_tMOTOR_s10_c0.3.pickle')
groups, _ = load_pickle(KNOCK_DIR, 'GaussianKO_mapping_tMOTOR_s10_c0.3.pickle')
params = load_pickle(KNOCK_DIR, 'DeepKO_params_DeepKO_tMOTOR_s10')

d = knockoff_class.DeepKnockOff('MOTOR', 10)
#d.pre_process(max_corr=.3, save=True)
#d.fit()
d.load_x(x_train)
d.load_params(params)
d.load_machine()
# d.diagnostics()
data_deep = d.transform(groups=groups)
ko_betas = d.statistic(data_deep, save=True)
uncorrected_betas, corrected_betas = d.threshold(ko_betas, save=True)
