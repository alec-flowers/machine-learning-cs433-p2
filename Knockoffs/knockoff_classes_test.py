from Knockoffs import knockoff_class
import pickle
k = knockoff_class.LowRankKnockOff('MOTOR', 0)
k.fit()
data = k.transform(save=True)

# k_so = knockoff_class.GaussianKnockOff('MOTOR', 0)
# k_so.pre_process(max_corr=.3, save=True)
#
# with open('../Data/GaussianKO_tfMRI_tMOTOR_s0_c0.3', "rb") as f:
#     SigmaHat, _ = pickle.load(f)
# k_so.fit(sigma_hat=SigmaHat)