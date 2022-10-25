import analysis_util_functions as uf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython import embed


def query_data_frame_to_get_index(f_df, f_tags):
    tags_idx_bool = []
    for col in f_tags:
        # check if col names exists in data frame
        if col not in f_df:
            uf.msg_box('WARNING', f'Column name "{col}" does not exist', sep='-')
            return None
        evaluation_string = f'f_df[col]{f_tags[col]}'
        f_idx = eval(evaluation_string)
        tags_idx_bool.append(f_idx)

    # Find Entries that are True for all tags
    tags_idx_bool_combined = np.all(tags_idx_bool, axis=0)
    # Find all Rows Index Numbers that are True
    tags_idx_index = np.where(tags_idx_bool_combined == True)[0]
    return tags_idx_bool_combined, tags_idx_index


def cut_out_windows(f_data, f_idx, f_before_secs, f_after_secs, selected_col):
    # Get sampling rate of first recording and assume others have the same ...
    f_fr = f_data.iloc[f_idx[0]]['fr']
    f_values = [[]] * len(f_idx)
    for f_k, f_s in enumerate(f_idx):
        f_before_samples = int(f_before_secs * f_fr)
        f_after_samples = int(f_after_secs * f_fr)
        f_val = df.iloc[f_s - f_before_samples:f_s + f_after_samples][selected_col]
        f_values[f_k] = f_val
    return f_values, f_fr


# Select Data File
file_dir = uf.select_file([('CSV Files', '.csv')])
df = pd.read_csv(file_dir, index_col=0)

# Set Tags
tags = {'stimulus_onset_type': '=="Step"', 'anatomy': '=="tg"', 'score': '>=0.6'}
idx_bool, idx_index = query_data_frame_to_get_index(df, tags)

# Cut out responses
responses, fr = cut_out_windows(df, idx_index, f_before_secs=5, f_after_secs=20, selected_col='z-score')
cell_names = df.loc[idx_bool]['id'].unique()

# Compute Mean and STD
# responses = np.array(responses)
m = np.mean(responses, axis=0)
sem = np.std(responses, axis=0) / np.sqrt(len(m))
t_axis = uf.convert_samples_to_time(sig=m, fr=fr)

# for kk in responses:
#     plt.plot(kk)
#     plt.show()

plt.figure()
plt.title(f'n={cell_names.shape[0]}')
plt.plot(t_axis, m, 'k')
plt.plot(t_axis, m-sem, 'r')
plt.plot(t_axis, m+sem, 'r')
plt.show()

embed()
exit()

time_before = 5
time_after = 20

idx = (df['stimulus_onset_type'] == 'Step') \
      & (df['stimulus_parameter'] == 100) \
      & (df['anatomy'] == 'tg') \
      & (df['mean_score'] > 0.2)

idx_ramp = np.where(idx == True)[0]
cell_names = df.loc[idx]['id'].unique()
z_scores = [[]] * len(idx_ramp)
fr = df.iloc[idx_ramp[0]]['fr']
for k, s in enumerate(idx_ramp):
    before = int(time_before * fr)
    after = int(time_after * fr)
    z = df.iloc[s-before:s+after]['z-score']
    z_scores[k] = z

plt.figure()
plt.title(f'n={cell_names.shape[0]}')
m = np.mean(z_scores, axis=0)
sem = np.std(z_scores, axis=0) / np.sqrt(len(m))
t_axis = uf.convert_samples_to_time(sig=m, fr=fr)
plt.plot(t_axis, m, 'k')
plt.plot(t_axis, m-sem, 'r')
plt.plot(t_axis, m+sem, 'r')
plt.show()