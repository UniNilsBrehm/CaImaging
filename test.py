from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import numpy as np


def exp_func(t, tau):
    y = np.exp(-t / tau)
    return y


def model_onset(tau=200, phase=300, n=700):
    # pre-allocatre memory with empty array of zeros
    onset_model = np.zeros(n)
    onset_model2 = np.zeros(n)

    decay = lambda t: np.exp(-t / tau)  # exponential decay

    vfunc = np.vectorize(decay)
    vfunc2 = np.vectorize(exp_func)

    # in our experiment LED turns on at frame phase = 300
    onset_model[phase:] = vfunc(np.array(range(phase, n)))
    onset_model2[phase:] = vfunc2(np.array(range(phase, n)), tau)

    return onset_model


# 3 stimulus types and 20 neurons with a score between 0 and 10
# X = [list(np.random.randint(0, 10, 20)), list(np.random.randint(0, 10, 20)), list(np.random.randint(0, 10, 20))]
# X = np.array([np.random.randint(0, 10, 20),  np.random.randint(0, 10, 20), np.random.randint(0, 10, 20)])
X = []
n_stimulus_types = 9
n_neurons = 20
min_score = 0
max_score = 10
for i in range(n_stimulus_types):
    X.append(np.random.randint(min_score, max_score, n_neurons))
X = np.array(X)

Z = linkage(X, 'ward')
# Z = linkage(X, 'single')

fig = plt.figure()
dn = dendrogram(Z)
plt.show()

####
# X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
clustering = AgglomerativeClustering(n_clusters=n_stimulus_types).fit(X)
