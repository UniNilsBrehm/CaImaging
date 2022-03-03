import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import scipy as sp


def zscore_example(means, sds, n):
    # Create Demo Data for two neurons
    # Inputs: means = [neuron1, neuron2]
    # Inputs: sds = [neuron1, neuron2]
    neuron1_mean = means[0]
    neuron1_sd = sds[0]
    neuron1 = np.random.normal(loc=neuron1_mean, scale=neuron1_sd, size=n)

    neuron2_mean = means[1]
    neuron2_sd = sds[1]
    neuron2 = np.random.normal(loc=neuron2_mean, scale=neuron2_sd, size=n)

    z_score1 = (neuron1 - neuron1_mean) / neuron1_sd
    z_score2 = (neuron2 - neuron2_mean) / neuron2_sd

    return neuron1, neuron2, z_score1, z_score2


bins = 100
n = 1000000
fig, (ax1, ax2, ax3) = plt.subplots(3, 3)
# First Row:
n1, n2, z1, z2 = zscore_example(means=[100, 100], sds=[8, 16], n=n)
ax1[0].hist(n1, bins=bins, histtype='step', color='blue', density=True)
ax1[0].hist(n2, bins=bins, histtype='step', color='green', density=True)
ax1[1].hist(z1, bins=bins, histtype='step', color='blue', density=True)
ax1[1].hist(z2, bins=bins, histtype='step', color='green', density=True)
ax1[2].plot(n1, z1, color='blue')
ax1[2].plot(n2, z2, color='green')
ax1[2].plot([100, 100], [np.min([z1, z2]), np.max([z1, z2])], 'r--', linewidth=0.5)
ax1[2].plot([np.min([n1, n2]), np.max([n1, n2])], [0, 0], 'k--', linewidth=0.5)
ax1[0].plot([100, 100], [0, 0.05], 'r--', linewidth=0.5)
ax1[0].set_title('Activity')
ax1[1].set_title('Z Score')
ax1[2].set_title('Activity vs. ZScore')
ax1[2].set_ylabel('Z Score')
ax1[0].legend(['Mean', 'Neuron 1', 'Neuron 2'], loc=2, prop={'size': 6})
ax1[0].set_yticks([])
ax1[1].set_yticks([])
ax1[0].set_xticks([0, 50, 100, 150, 200])
ax1[2].set_xticks([0, 50, 100, 150, 200])
ax1[0].set_xlim([0, 200])
ax1[2].set_xlim([0, 200])

# Second Row:
n1, n2, z1, z2 = zscore_example(means=[100, 120], sds=[8, 8], n=n)
ax2[0].hist(n1, bins=bins, histtype='step', color='blue', density=True)
ax2[0].hist(n2, bins=bins, histtype='step', color='green', density=True)
ax2[1].hist(z1, bins=bins, histtype='step', color='blue', density=True)
ax2[1].hist(z2, bins=bins, histtype='step', color='green', density=True)
ax2[2].plot(n1, z1, color='blue')
ax2[2].plot(n2, z2, color='green')
ax2[2].plot([np.min([n1, n2]), np.max([n1, n2])], [0, 0], 'k--', linewidth=0.5)
ax2[2].plot([100, 100], [np.min([z1, z2]), np.max([z1, z2])], 'b--', linewidth=0.5)
ax2[2].plot([120, 120], [np.min([z1, z2]), np.max([z1, z2])], 'g--', linewidth=0.5)
ax2[0].plot([100, 100], [0, 0.05], 'b--', linewidth=0.5)
ax2[0].plot([120, 120], [0, 0.05], 'g--', linewidth=0.5)
ax2[2].set_ylabel('Z Score')
ax2[0].set_yticks([])
ax2[1].set_yticks([])
ax2[0].set_xticks([0, 50, 100, 150, 200])
ax2[2].set_xticks([0, 50, 100, 150, 200])
ax2[0].set_xlim([0, 200])
ax2[2].set_xlim([0, 200])

# Third Row:
n1, n2, z1, z2 = zscore_example(means=[100, 120], sds=[8, 16], n=n)
ax3[0].hist(n1, bins=bins, histtype='step', color='blue', density=True)
ax3[0].hist(n2, bins=bins, histtype='step', color='green', density=True)
ax3[1].hist(z1, bins=bins, histtype='step', color='blue', density=True)
ax3[1].hist(z2, bins=bins, histtype='step', color='green', density=True)
ax3[2].plot(n1, z1, color='blue')
ax3[2].plot(n2, z2, color='green')
ax3[0].plot([100, 100], [0, 0.05], 'b--', linewidth=0.5)
ax3[0].plot([120, 120], [0, 0.05], 'g--', linewidth=0.5)
ax3[2].plot([100, 100], [np.min([z1, z2]), np.max([z1, z2])], 'b--', linewidth=0.5)
ax3[2].plot([120, 120], [np.min([z1, z2]), np.max([z1, z2])], 'g--', linewidth=0.5)
ax3[2].plot([np.min([n1, n2]), np.max([n1, n2])], [0, 0], 'k--', linewidth=0.5)
ax3[2].set_xlabel('Activity')
ax3[1].set_xlabel('Z Score')
ax3[0].set_xlabel('Activity')
ax3[0].set_yticks([])
ax3[1].set_yticks([])
ax3[2].set_ylabel('Z Score')
ax3[0].set_xticks([0, 50, 100, 150, 200])
ax3[2].set_xticks([0, 50, 100, 150, 200])
ax3[0].set_xlim([0, 200])
ax3[2].set_xlim([0, 200])

# Save Fig
plt.tight_layout()
plt.savefig('zscore_demo.pdf')
plt.close(fig)
print('ZScore Demo Fig saved!')
