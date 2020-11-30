import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

PATH = '../GLM/betas/betas_knockoff_test.pickle'
PATH2 = './GLM/betas/tvalues_knockoff_test.pickle'
with open(PATH, 'rb') as pickle_file:
    betas = pickle.load(pickle_file)

with open(PATH2, 'rb') as pickle_file:
    tvalues = pickle.load(pickle_file)

fig, axs = plt.subplots(1, 2, figsize=(11, 8))

region = 0
task = 0
sns.histplot(x=betas[:, region, task], bins=15, ax=axs[0])
axs[0].axvline(x=betas[0, region, task], color='r')

sns.histplot(x=tvalues[:, region, task], bins=15, ax=axs[1])
axs[1].axvline(x=tvalues[0, region, task], color='r')
plt.show()