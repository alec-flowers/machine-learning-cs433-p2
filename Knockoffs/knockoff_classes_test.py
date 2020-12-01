from Knockoffs import knockoff_class
import pickle
# k = knockoff_class.LowRankKnockOff('MOTOR', 0)
# k.fit()
# data = k.transform(save=True)

k_so = knockoff_class.GaussianKnockOff('MOTOR', 0)
k_so.pre_process(max_corr=.3, save=True)
k_so.fit()
data_so = k_so.transform()
pass