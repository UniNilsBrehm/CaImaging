import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from collections import Counter
from IPython import embed
import time as stop_watch
from PIL import Image


def plot_average_responses(_clust_amount, _data, _labels_sorted, _row_labels, _number_of_stim, _title_label,
                           y_label):
    _fig, _axs = plt.subplots(_clust_amount, _number_of_stim, sharey=True)
    for ii in np.arange(_clust_amount):
        for jj in np.arange(_number_of_stim):
            mean = np.mean(_data[jj, np.where(_labels_sorted == ii)[1], :], axis=0)
            _axs[ii, jj].plot(mean, color='black')
            sd = np.std(_data[jj, np.where(_labels_sorted == ii)[1], :], axis=0)
            _axs[ii, jj].plot(mean + sd, color='red', alpha=0.2)
            _axs[ii, jj].plot(mean - sd, color='red', alpha=0.2)
            _axs[_clust_amount - 1, jj].set_xlabel(_row_labels[jj])
            _axs[ii, 0].set_ylabel(y_label + str(ii + 1))
    _fig.suptitle(_title_label, fontsize=16)
    return 0


t1 = stop_watch.perf_counter()
dir_name_save = 'C:/Uni Freiburg/NilsWenke_Datenauswertung_2021/Raw_data/Results/'
allFiles = os.listdir(dir_name_save)

# Loads in all data and saves it in an array ---------------------------------------------------------------------------
combined_data = np.zeros((6, 1))
cutout_data = np.zeros((30, 1, 50))
zscore_data = np.zeros((1, 3000))

for txtFile in allFiles:
    if txtFile.endswith("_Scores.txt"):

        scorename = dir_name_save + txtFile  # filename to be processed
        single_data = np.transpose(np.loadtxt(scorename))
        if single_data.size != 0:
            combined_data = np.append(combined_data, single_data, axis=1)

    elif txtFile.endswith("_cutouts.txt"):

        cutoutname = dir_name_save + txtFile  # filename to be processed
        single_data = np.loadtxt(cutoutname)

        if single_data.size != 0:
            single_data_cut = np.reshape(single_data, (30, int(single_data.shape[0] / 30), 50))

            cutout_data = np.append(cutout_data, single_data_cut, axis=1)

    elif txtFile.endswith("_zscore.txt"):
        zscorename = dir_name_save + txtFile  # filename to be processed
        single_data = np.loadtxt(zscorename)
        if single_data.size != 0:
            data_frame = np.zeros((single_data.shape[1], 3000))
            data_frame[0:single_data.shape[1], 0:len(single_data)] = np.transpose(single_data)
            zscore_data = np.append(zscore_data, data_frame, axis=0)

combined_data = np.delete(combined_data, (0), axis=1)
cutout_data = np.delete(cutout_data, (0), axis=1)
zscore_data = np.delete(zscore_data, (0), axis=0)

# Clustering with hierachical Algorithm---------------------------------------------------------------------------------
# Number of clusters:
clust_amount = 5

# scipy.cluster.hierarchy.linkage performance clustering
# scipy.cluster.hierarchy.dendrogram takes the output from this clustering and computes dendrogram plot
dendrogram = sch.dendrogram(sch.linkage(np.transpose(combined_data), method="ward"))
plt.title('Dendrogram')
plt.ylabel('Euclidean distances')
plt.show()

# Performing Agglomerative Clustering (=Dendrogram) using sk-learn package
hc = AgglomerativeClustering(n_clusters=clust_amount, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(np.transpose(combined_data))

# Sorts the Scores by cluster
inds = y_hc.argsort()  # returns indices that would sort this array
combined_data_sorted = combined_data[:, inds]
labels_sorted = np.zeros((1, combined_data_sorted[0:6, :].shape[1]))
labels_sorted[0, :] = y_hc[inds]
cutout_data_sorted = cutout_data[:, inds, :]

# FIGURE: Sorts by cluster
figure, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 8]})
heatmap = a1.pcolor(combined_data_sorted[0:6, :], vmin=0, vmax=3)
heatmap2 = a0.pcolor(labels_sorted, cmap='gray')
a1.set_yticks(np.arange(combined_data_sorted[0:6, :].shape[0]) + 0.5, minor=False)
row_labels = ['Step', 'Ramp 1', 'Ramp 2', 'Ramp 3', 'Ramp 4', 'Ramp 5']
a1.set_yticklabels(row_labels, minor=False)
a0.set_title('Clusters')
a1.set_xlabel('Cells #')
plt.show()

# --- AVERAGE RESPONSES ------------------------------------------------------------------------------------------------
cutout_data_mean = np.mean(np.reshape(cutout_data_sorted, (6, 5, 100, 50)), axis=1)
# FIGURE
# plot_average_responses(_clust_amount=clust_amount, _row_labels=row_labels, _labels_sorted=labels_sorted,
#                        _data=cutout_data_mean, _number_of_stim=6, _title_label='Cluster responses')
fig, axs = plt.subplots(clust_amount, 6, sharey=True)
for ii in np.arange(clust_amount):
    for jj in np.arange(6):
        mean = np.mean(cutout_data_mean[jj, np.where(labels_sorted == ii)[1], :], axis=0)
        axs[ii, jj].plot(mean, color='black')
        sd = np.std(cutout_data_mean[jj, np.where(labels_sorted == ii)[1], :], axis=0)
        axs[ii, jj].plot(mean + sd, color='red', alpha=0.2)
        axs[ii, jj].plot(mean - sd, color='red', alpha=0.2)
        axs[clust_amount-1, jj].set_xlabel(row_labels[jj])
        axs[ii, 0].set_ylabel('Cluster ' + str(ii+1))
fig.suptitle("Cluster responses", fontsize=16)
plt.show()

t2 = stop_watch.perf_counter()
print(f'Clustering took: {t2-t1} secs')

# --- MANUAL ANATOMY REGISTRATION OF ALL 100 RESPONSIVE CELLS ----------------------------------------------------------
# +++ Nisse +++ Dont have this file ...
dir_name = 'C:/Users/nilsw/Documents/Dokumente Nils/Arbeit und Bewerbungen/Jobs/Biologie 1/Bio_Melanie/Data/notebooks/Final_Notebooks/Original_Data/'

ref_name = 'reference.tif'
ref = dir_name + ref_name
# reference = Image.open(ref)
# reference_array= np.array(reference)
# plt.imshow(reference_array)

# Manually labeled anatomy:
anatomy_labels = ['Not clear','ALLG','TG','PTN','PTP','PTac','PTar']
anatomy = np.array([1,2,2,2,2,2,1,2,1,2,5,5,0,2,2,2,1,1,1,5,0,0,5,5,5,3,3,2,2,2,2,4,4,4,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,1
                       ,6,6,1,1,1,2,2,2,2,0,0,4,5,6,0,0,0,0,0,0,0,0,0,2,2,2,2,2,0,0,2,2,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,
                    2,2,2])

# Sorts the Scores by anatomy
inds = anatomy.argsort()
combined_data_sorted = combined_data[:, inds]
labels_sorted = np.zeros((1, combined_data_sorted[0:6, :].shape[1]))
labels_sorted[0, :] = anatomy[inds]
cutout_data_sorted = cutout_data[:, inds, :]

# Sorts by cluster
figure, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 8]})

heatmap = a1.pcolor(combined_data_sorted[0:6, :], vmin=0, vmax=3)
heatmap2 = a0.pcolor(labels_sorted, cmap='gray')


a1.set_yticks(np.arange(combined_data_sorted[0:6, :].shape[0]) + 0.5, minor=False)
row_labels = ['Step', 'Ramp 1', 'Ramp 2', 'Ramp 3', 'Ramp 4', 'Ramp 5']
a1.set_yticklabels(row_labels, minor=False)
a0.set_title('Cell types')
a1.set_xlabel('Cells #')
plt.show()

cutout_data_mean = np.mean(np.reshape(cutout_data_sorted, (6, 5, 100, 50)), axis=1)
anatomies = len(set(anatomy))

fig, axs = plt.subplots(anatomies, 6, sharey=True)
for ii in np.arange(anatomies):
    for jj in np.arange(6):
        axs[ii, jj].plot(np.mean(cutout_data_mean[jj, np.where(labels_sorted == ii)[1], :], axis=0))
        axs[anatomies-1, jj].set_xlabel(row_labels[jj])
        axs[ii, 0].set_ylabel(anatomy_labels[ii])
fig.suptitle("Anatomical responses", fontsize=16)
plt.show()

# Plot single traces:
# zscore_data_sorted = zscore_data[inds,:]
# AllG_cells = np.where(labels_sorted==5)[1]
# i = 0
#
# for cells in AllG_cells:
#     plt.figure(i)
#     plt.plot(zscore_data_sorted[cells,:])
#     i = i+1
print('Finished!')
