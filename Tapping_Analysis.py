import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import util_functions as uf
import pickle
import seaborn as sns


def plot_raw_f(data, meta, f_rois, x_lim, data_type):
    roi_count = len(rois)
    t_rois = uf.samples_to_time(data.loc[data['roi'] == f_rois[0]][data_type], meta['sampling_rate'])
    f_fig, f_axs = plt.subplots(roi_count+1, 1, figsize=(5, 8), sharex=True)  # fig size = (width, height)
    if len(x_lim) > 1:
        data_to_plot = data.loc[(data['time'] < x_lim[1]) & (data['time'] >= x_lim[0])]
        f_axs[0].set_xlim([x_lim[0], x_lim[1]])
    else:
        data_to_plot = data

    f_axs[0].plot(meta['stimulation']['Time'], meta['stimulation']['Volt'], 'k', linewidth=0.5)
    y_lim = [np.min(data_to_plot[data_type]), np.max(data_to_plot[data_type])]
    for kk in range(roi_count):
        # plotting_data = data.loc[data['roi'] == f'Mean{f_rois[kk]}'][data_type]
        plotting_data = data.loc[data['roi'] == f_rois[kk]][data_type]

        f_axs[kk+1].plot(t_rois, plotting_data, 'b', linewidth=0.5)
        if len(x_lim) > 1:
            f_axs[kk+1].set_xlim([x_lim[0], x_lim[1]])
        f_axs[kk+1].set_ylim(y_lim)

    sns.despine(top=True, right=True, left=True, bottom=True, offset=2)
    plt.show()


experiment_type = 'Tapping'
base_dir = 'C:/Uni Freiburg/CaImagingAnalysis/Tapping'
date_dir = '20220203'
recording_id = '01'
dir_path = f'{base_dir}/{date_dir}/{recording_id}'
protocol_path = f'{dir_path}/logs/protocols/'
static_path = f'{dir_path}/logs/statics/'
stimulation_path = f'{dir_path}/stimulation/'
rawdata_path = f'{dir_path}/rawdata/'
reference_path = f'{dir_path}/references/'

# Load data
d = pd.read_csv(f'{dir_path}/data_long.csv')
d = d.drop([d.keys()[0]], axis=1)

# Load temp from HDD
metadata_file_name = f'{dir_path}/MetaData.pkl'
open_file = open(metadata_file_name, "rb")
metadata = pickle.load(open_file)
open_file.close()

#
s_parameter = 1600
s_type = 's'
roi = 'Mean1'
rois = d['roi'].unique()
p_data = d.loc[(d['parameter'] == s_parameter) & (d['roi'] == roi) & (d['type'] == s_type)]
embed()
exit()
plot_raw_f(data=d, meta=metadata, f_rois=rois, x_lim=[], data_type='deltaf')
t_rois = uf.samples_to_time(d.loc[d['roi'] == 'Mean1']['deltaf'], metadata['sampling_rate'])
n = 0
for r in rois:
    n = n + 1
    plt.figure(n)
    plt.title(f'Roi: {n}')
    plt.plot(metadata['stimulation']['Time'], metadata['stimulation']['Volt'], 'k')
    plt.plot(t_rois, d.loc[d['roi'] == r]['zscore'], 'b')
    plt.show()

#
good_rois = [1, 2, 3, 4, 5, 6]
for k in good_rois:
    a = d.loc[(d['roi'] == f'Mean{k}') & (d['type'] == 's')]
# sns.relplot(x="trial_time", y="deltaf", hue="parameter", style="parameter",
#             kind="line", data=d)
#
# sns.lineplot(
#     data=d.query("roi == 'Mean1'"),
#     x="trial_time", y="deltaf", hue="type"
# )

embed()
exit()
