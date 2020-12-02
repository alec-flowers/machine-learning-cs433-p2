from Knockoffs import knockoff_class
from Knockoffs.params import get_params
from input_output import load
import pickle

#Example of running LowRankKnockOff
k = knockoff_class.LowRankKnockOff('MOTOR', 0)
k.load_fmri()
k.fit()
#k.diagnostics()
data = k.transform(save=True)

#Example of running GaussianKnocckOff
k_so = knockoff_class.GaussianKnockOff('MOTOR', 0)
k_so.load_fmri()
k_so.pre_process(max_corr=.3)
corr_g = k_so.fit()
#k_so.diagnostics()
data_so = k_so.transform()

#Example of running DeepKnockOff
d = knockoff_class.DeepKnockOff('MOTOR', 0)
d.pre_process(max_corr=.3)
# d.fit()
d.load_machine()
#d.diagnostics()
data_deep = d.transform()
