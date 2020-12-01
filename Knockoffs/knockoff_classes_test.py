from Knockoffs import knockoff_class

k = knockoff_class.LowRankKnockOff('MOTOR', 1)
k.fit()
data = k.transform()
