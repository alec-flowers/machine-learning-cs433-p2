import numpy as np
import matplotlib.pyplot as plt
import pickle

PATH = './GLM/betas/betas_knockoff_test.pickle'
with open(PATH, 'rb') as pickle_file:
    betas = pickle.load(pickle_file)

fig, axs = plt.subplots(1, figsize=(11, 8))

region = 0
task = 0
axs.hist(betas[:, region, task])
axs.axvline(x=betas[0, region, task], color='r')
plt.show()