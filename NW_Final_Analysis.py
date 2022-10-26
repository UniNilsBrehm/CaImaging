import analysis_util_functions as uf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython import embed
from scipy.stats import median_abs_deviation
import Estimate_GCaMP_tau as estimate_tau


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


def plot_response(f_data, f_tags, f_window, activity_measure='z-score'):
    f_idx_bool, f_idx_index = query_data_frame_to_get_index(f_data, f_tags)

    # Cut out responses
    f_responses, f_fr = cut_out_windows(
        f_data, f_idx_index, f_before_secs=f_window[0], f_after_secs=f_window[1], selected_col=activity_measure
    )
    f_cell_names = f_data.loc[f_idx_bool]['id'].unique()

    # Compute Mean and STD
    # f_m = np.mean(f_responses, axis=0)
    f_m = np.median(f_responses, axis=0)
    # f_sem = np.std(f_responses, axis=0) / np.sqrt(len(f_m))
    f_sem = median_abs_deviation(f_responses, axis=0)
    f_t_axis = uf.convert_samples_to_time(sig=f_m, fr=f_fr) - f_window[0]

    cell_group = f_tags['anatomy'][2:]
    f_stimulus = f_tags['stimulus_onset_type'][2:]
    f_score = f_tags['score'][2:]

    plt.figure()
    for kk in f_responses:
        plt.plot(f_t_axis, kk, 'k', lw=0.1)

    plt.title(f'{cell_group}, cells={f_cell_names.shape[0]}, n={len(f_responses)}, th: {f_score}, {f_stimulus}')
    plt.plot(f_t_axis, f_m, 'k')
    plt.plot(f_t_axis, f_m - f_sem, 'r')
    plt.plot(f_t_axis, f_m + f_sem, 'r')
    plt.xlabel('Time [s]')
    plt.ylabel(f'{activity_measure}')
    plt.ylim([-1, 2])
    # plt.show()


# Select Data File
file_dir = uf.select_file([('CSV Files', '.csv')])
df = pd.read_csv(file_dir, index_col=0)
embed()
exit()
before = 5
after = 25
th_score = 0.01

# tags = {'stimulus_onset_type': '=="Step"', 'anatomy': '=="allg"', 'score': f'>={th_score}'}
tags = {'anatomy': '=="allg"', 'score': f'>={th_score}'}

idx_bool, idx_index = query_data_frame_to_get_index(df, tags)
responses, fr = cut_out_windows(df, idx_index, f_before_secs=before, f_after_secs=after, selected_col='df')
t_samples = np.arange(0, len(responses[0]), 1)

tau_e = []
err_e = []
for re in responses:
    popt, pcov = estimate_tau.fit_exp(x=t_samples, y=re, full_exp=True, plot_results=False)
    tau = popt[1]
    p_err = np.sqrt(np.diag(pcov))
    err_tau = p_err[1]
    tau_e.append(tau)
    err_e.append(err_tau)

err_e = np.array(err_e)
tau_e = np.array(tau_e)
idx = (err_e < 100) & (tau_e < 100)
err_e = err_e[idx]
tau_e = tau_e[idx]
print(tags)
print(f'tau: {np.round(np.mean(tau_e * 2), 2)} s (+- {np.round(np.mean(err_e* 2), 2)} SD), n={len(tau_e)}')

# Set Tags

before = 5
after = 25
th_score = 0.1
tags = {'stimulus_onset_type': '=="Ramp"', 'anatomy': '=="allg"', 'score': f'>={th_score}'}
plot_response(f_data=df, f_tags=tags, f_window=[before, after], activity_measure='df')

tags = {'stimulus_onset_type': '=="Step"', 'anatomy': '=="allg"', 'score': f'>={th_score}'}
plot_response(f_data=df, f_tags=tags, f_window=[before, after], activity_measure='df')
plt.show()

exit()
idx_bool, idx_index = query_data_frame_to_get_index(df, tags)

# Cut out responses
responses, fr = cut_out_windows(df, idx_index, f_before_secs=3, f_after_secs=20, selected_col='z-score')
cell_names = df.loc[idx_bool]['id'].unique()

# Compute Mean and STD
# responses = np.array(responses)
m = np.mean(responses, axis=0)
sem = np.std(responses, axis=0) / np.sqrt(len(m))
t_axis = uf.convert_samples_to_time(sig=m, fr=fr) - 3

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