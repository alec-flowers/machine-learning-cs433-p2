import knockoff_class

# Example of running LowRankKnockOff

k = knockoff_class.LowRankKnockOff('MOTOR', 0)
k.load_fmri()
k.fit(rank=120)
k.diagnostics()
data = k.transform()
ko_betas = k.statistic(data, save=True)
uncorrected_betas, corrected_betas = k.threshold(ko_betas, save=True)

# Example of running GaussianKnocckOff

k_so = knockoff_class.GaussianKnockOff('MOTOR', 0)
k_so.load_fmri()
k_so.pre_process(max_corr=.3)
corr_g = k_so.fit()
k_so.diagnostics()
data_so = k_so.transform()
ko_betas = k_so.statistic(data_so, save=True)
uncorrected_betas, corrected_betas = k_so.threshold(ko_betas, save=True)

#
# # Example of running DeepKnockOff
#
d = knockoff_class.DeepKnockOff('MOTOR', 0)
d.pre_process(max_corr=.3)
d.fit()
# d.load_machine()
d.diagnostics()
data_deep = d.transform()
ko_betas = d.statistic(data_deep, save=True)
uncorrected_betas, corrected_betas = d.threshold(ko_betas, save=True)
