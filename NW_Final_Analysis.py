import analysis_util_functions as uf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython import embed
from scipy.stats import median_abs_deviation
import Estimate_GCaMP_tau as estimate_tau
import pickle


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


def estimate_tau_values(f_data, f_tags):

    t_samples = np.arange(0, len(f_data[0]), 1)

    tau_e = []
    err_e = []
    for re in f_data:
        popt, pcov = estimate_tau.fit_exp(x=t_samples, y=re, full_exp=True, plot_results=False)
        tau = popt[1]
        p_err = np.sqrt(np.diag(pcov))
        err_tau = p_err[1]
        tau_e.append(tau)
        err_e.append(err_tau)

    err_e = np.array(err_e)
    tau_e = np.array(tau_e)
    f_idx = (err_e < 100) & (tau_e < 100)
    err_e = err_e[f_idx]
    tau_e = tau_e[f_idx]
    print(f_tags)
    print(f'tau: {np.round(np.mean(tau_e * 2), 2)} s (+- {np.round(np.mean(err_e * 2), 2)} SD), n={len(tau_e)}')


def plot_response(f_data, f_tags, f_window, activity_measure='z-score'):
    f_idx_bool, f_idx_index = query_data_frame_to_get_index(f_data, f_tags)

    # Cut out responses
    f_responses, f_fr = cut_out_windows(
        f_data, f_idx_index, f_before_secs=f_window[0], f_after_secs=f_window[1], selected_col=activity_measure
    )
    f_cell_names = f_data.loc[f_idx_bool]['id'].unique()

    # Compute Mean and STD
    f_m = np.mean(f_responses, axis=0)
    # f_m = np.median(f_responses, axis=0)
    f_sem = np.std(f_responses, axis=0) / np.sqrt(len(f_m))
    # f_sem = median_abs_deviation(f_responses, axis=0)
    f_t_axis = uf.convert_samples_to_time(sig=f_m, fr=f_fr) - f_window[0]

    cell_group = f_tags['anatomy'][2:]
    f_stimulus = f_tags['stimulus_onset_type'][2:]
    # f_score = f_tags['score'][2:]

    plt.figure()
    for kk in f_responses:
        plt.plot(f_t_axis, kk, 'k', lw=0.1)

    plt.title(f'{cell_group}, cells={f_cell_names.shape[0]}, n={len(f_responses)}, {f_stimulus}')
    plt.plot(f_t_axis, f_m, 'k')
    plt.plot(f_t_axis, f_m - f_sem, 'r')
    plt.plot(f_t_axis, f_m + f_sem, 'r')
    plt.xlabel('Time [s]')
    plt.ylabel(f'{activity_measure}')
    plt.ylim([-1, 6])
    # plt.show()


def get_trials_for_cells(f_data, f_tags, f_window):
    # f_window = [5, 25]
    activity_measure = 'z-score'
    # Find entries
    f_idx_bool, f_idx_index = query_data_frame_to_get_index(f_data, f_tags)
    selected_data = f_data[f_idx_bool].copy()
    # Get Cell IDs
    f_cell_names = selected_data['id'].unique()
    # Go through each cell and get the trials with scores above threshold
    f_cell_responses = dict()
    # f_cell_means = []
    f_cell_means = dict()
    for c_name in f_cell_names:
        # Cut out responses
        f_idx_cell = selected_data['id'] == c_name
        cell_data = selected_data[f_idx_cell]
        f_index_cell = cell_data.index
        f_responses, f_fr = cut_out_windows(
            f_data, f_index_cell, f_before_secs=f_window[0], f_after_secs=f_window[1], selected_col=activity_measure
        )
        # Check for empty entries
        f_responses_checked = []
        count_empty_entries = 0
        for entry in f_responses:
            if entry.empty:
                count_empty_entries += 1
            else:
                f_responses_checked.append(entry)
        if count_empty_entries >= len(f_responses_checked):
            print('ERROR: COULD NOT FIND ANY RESPONSES')

        f_cell_responses[c_name] = f_responses_checked
        f_cell_means[c_name] = np.mean(f_responses_checked, axis=0)
        # f_cell_means.append(np.mean(f_responses, axis=0))
    return f_cell_responses, f_cell_means


def get_cell_grand_average_score(_data):
    # Get all cell names
    _cell_names = _data['id'].unique()
    grand_average_score = dict()
    for c in _cell_names:
        idx1 = _data['id'] == c
        idx2 = _data['mean_score'][idx1] > 0
        grand_average_score[c] = np.round(_data['score'].loc[idx1][idx2].mean(), 4)
    grand_average_score = pd.Series(grand_average_score)
    return grand_average_score


def get_data_for_matrix_plots(data, save_path):
    f_tags = [
        {'stimulus_onset_type': '=="Step"', 'anatomy': '=="tg"'},
        {'stimulus_onset_type': '=="Ramp"', 'anatomy': '=="tg"'},
        {'stimulus_onset_type': '=="Step"', 'anatomy': '=="allg"'},
        {'stimulus_onset_type': '=="Ramp"', 'anatomy': '=="allg"'},
    ]
    f_labels = ['tg_step', 'tg_ramp', 'allg_step', 'allg_ramp']

    for kk, vv in enumerate(f_tags):
        f_cell_trials, f_cell_means = get_trials_for_cells(f_data=data, f_tags=f_tags[kk])
        f_cell_means = pd.DataFrame(f_cell_means)
        # Store to HDD
        with open(f'{save_path}/matrix_plot_{f_labels[kk]}_all_trials.pkl', 'wb') as f:
            pickle.dump(f_cell_trials, f)

        # with open('saved_dictionary.pkl', 'rb') as f:
        #     loaded_dict = pickle.load(f)
        f_cell_means.to_csv(f'{save_path}/matrix_plot_{f_labels[kk]}.csv', index=False)
    uf.msg_box('INFO', 'Data for Matrix Plots stored to HDD (csv files)', '-')


def get_audio_cells_for_matrix_plots(data, save_path):
    f_tags = [
        {'stimulus_onset_type': '=="Step"', 'anatomy': '=="audio"'},
        {'stimulus_onset_type': '=="Ramp"', 'anatomy': '=="audio"'},
        {'stimulus_onset_type': '=="Step"', 'anatomy': '=="pllg"'},
        {'stimulus_onset_type': '=="Ramp"', 'anatomy': '=="pllg"'},
    ]
    f_labels = ['audio_step', 'audio_ramp', 'pllg_step', 'pllg_ramp']

    for kk, vv in enumerate(f_tags):
        f_cell_trials, f_cell_means = get_trials_for_cells(f_data=data, f_tags=f_tags[kk])
        f_cell_means = pd.DataFrame(f_cell_means)
        # Store to HDD
        with open(f'{save_path}/matrix_plot_{f_labels[kk]}_all_trials.pkl', 'wb') as f:
            pickle.dump(f_cell_trials, f)

        # with open('saved_dictionary.pkl', 'rb') as f:
        #     loaded_dict = pickle.load(f)
        f_cell_means.to_csv(f'{save_path}/matrix_plot_{f_labels[kk]}.csv', index=False)
    uf.msg_box('INFO', 'Audio and PLLG Cells for Matrix Plots stored to HDD (csv files)', '-')


def sound_get_data_for_matrix_plots(data, save_path, f_all_windows):
    # This is for Sound Stimuli
    # f_all_windows = ['SinglePulse', 'TrainPulse', 'FirstTrainPulse']
    # f_window = [before onset, after onset]
    f_tags = [
        # Single Pulses
        {'stimulus_onset_type': '=="SinglePulse"', 'stimulus_onset_parameter': '==0', 'anatomy': '=="audio"'},
        {'stimulus_onset_type': '=="SinglePulse"', 'stimulus_onset_parameter': '==0', 'anatomy': '=="allg"'},
        {'stimulus_onset_type': '=="SinglePulse"', 'stimulus_onset_parameter': '==0', 'anatomy': '=="tg"'},

        # Train Pulses from trains with interval of 5 secs
        {'stimulus_onset_type': '=="TrainPulse"', 'stimulus_onset_parameter': '==5', 'anatomy': '=="audio"'},
        {'stimulus_onset_type': '=="TrainPulse"', 'stimulus_onset_parameter': '==5', 'anatomy': '=="allg"'},
        {'stimulus_onset_type': '=="TrainPulse"', 'stimulus_onset_parameter': '==5', 'anatomy': '=="tg"'},

        # Train Pulses from trains with interval of 10 secs
        {'stimulus_onset_type': '=="TrainPulse"', 'stimulus_onset_parameter': '==10', 'anatomy': '=="audio"'},
        {'stimulus_onset_type': '=="TrainPulse"', 'stimulus_onset_parameter': '==10', 'anatomy': '=="allg"'},
        {'stimulus_onset_type': '=="TrainPulse"', 'stimulus_onset_parameter': '==10', 'anatomy': '=="tg"'},

        # Complete Trains interval of 5 secs
        {'stimulus_onset_type': '=="FirstTrainPulse"', 'stimulus_onset_parameter': '==5', 'anatomy': '=="audio"'},
        {'stimulus_onset_type': '=="FirstTrainPulse"', 'stimulus_onset_parameter': '==5', 'anatomy': '=="allg"'},
        {'stimulus_onset_type': '=="FirstTrainPulse"', 'stimulus_onset_parameter': '==5', 'anatomy': '=="tg"'},

        # Complete Trains interval of 10 secs
        {'stimulus_onset_type': '=="FirstTrainPulse"', 'stimulus_onset_parameter': '==10', 'anatomy': '=="audio"'},
        {'stimulus_onset_type': '=="FirstTrainPulse"', 'stimulus_onset_parameter': '==10', 'anatomy': '=="allg"'},
        {'stimulus_onset_type': '=="FirstTrainPulse"', 'stimulus_onset_parameter': '==10', 'anatomy': '=="tg"'}

    ]
    f_labels = [
        'audio_SinglePulse', 'allg_SinglePulse', 'tg_SinglePulse',
        'audio_TrainPulse_05', 'allg_TrainPulse_05', 'tg_TrainPulse_05',
        'audio_TrainPulse_10', 'allg_TrainPulse_10', 'tg_TrainPulse_10',
        'audio_FirstTrainPulse_05', 'allg_FirstTrainPulse_05', 'tg_FirstTrainPulse_05',
        'audio_FirstTrainPulse_10', 'allg_FirstTrainPulse_10', 'tg_FirstTrainPulse_10',
    ]

    f_windows = [
        f_all_windows[0], f_all_windows[0], f_all_windows[0],
        f_all_windows[1], f_all_windows[1], f_all_windows[1],
        f_all_windows[1], f_all_windows[1], f_all_windows[1],
        f_all_windows[2], f_all_windows[2], f_all_windows[2],
        f_all_windows[2], f_all_windows[2], f_all_windows[2]
    ]

    for kk, vv in enumerate(f_tags):
        f_cell_trials, f_cell_means = get_trials_for_cells(f_data=data, f_tags=f_tags[kk], f_window=f_windows[kk])
        f_cell_means = pd.DataFrame(f_cell_means)
        # Store to HDD
        with open(f'{save_path}/matrix_plot_{f_labels[kk]}_all_trials.pkl', 'wb') as f:
            pickle.dump(f_cell_trials, f)

        # with open('saved_dictionary.pkl', 'rb') as f:
        #     loaded_dict = pickle.load(f)
        f_cell_means.to_csv(f'{save_path}/matrix_plot_{f_labels[kk]}.csv', index=False)
    uf.msg_box('INFO', 'Audio and PLLG Cells for Matrix Plots stored to HDD (csv files)', '-')


def get_single_traces(data, selection_id):
    data_selected = data[data['id'] == selection_id].copy()
    df_f_selected = data_selected['df']
    cell_anatomy = data_selected['anatomy'].unique()[0]
    fr_rec = 2.0345147125756804
    time_axis = uf.convert_samples_to_time(sig=df_f_selected, fr=fr_rec)
    return time_axis, df_f_selected, cell_anatomy


def get_selected_traces(data, save_path, protocol_template):
    fr_rec = 2.0345147125756804
    onset_template = protocol_template['Onset_Time'].iloc[0]
    selection = ['220525_03_01_roi_1', '220525_03_01_roi_6', '220525_04_01_roi_1', '220525_04_01_roi_10']
    for sel in selection:
        t, df_f, c_anatomy = get_single_traces(data, sel)
        pp = pd.read_csv(f'{save_path}/{sel[:12]}_protocol.csv', index_col=0)
        onset = pp['Onset_Time'].iloc[0]
        onset_diff = onset - onset_template
        time_axis = uf.convert_samples_to_time(sig=df_f, fr=fr_rec)
        # align time axis to template
        time_axis = time_axis - onset_diff
        pd.DataFrame(np.array([time_axis, df_f]).T, columns=['time', 'df']).to_csv(f'{save_path}/example_trace_{sel}_{c_anatomy}.csv')
    print(f'Example Traces for {selection} stored to HDD')


def get_example_traces(data, save_path, protocol_template):
    # data[0]: tg and allg cells
    # data[1]: audio and pllg cells
    # base_path = 'E:/CaImagingAnalysis/Paper_Data/Figures/data'
    # f_stimulation = pd.read_csv(f'{base_path}/180417_2_1_stimulation_example_traces.txt')
    # protocol_template = pd.read_csv(f'{base_path}/180417_2_1_protocol.csv', index_col=0)
    onset_template = protocol_template['Onset_Time'].iloc[0]

    # Tapping Recs:
    f_recs = ['180417_4_1', '180417_2_1', '180417_2_1', '180418_2_1']
    f_labels = ['allg', 'tg', 'audio', 'pllg']
    data_set = [0, 0, 1, 1]
    cell_nr = [['roi_10', 'roi_18'], ['roi_14', 'roi_17'], ['roi_5', 'roi_8'], ['roi_14', 'roi_15']]
    fr_rec = 2.0345147125756804

    # Sound Recs:
    # f_recs = ['220525_03_01', '220525_04_01']
    # f_labels = ['allg', 'tg', 'audio']
    # data_set = [0, 1]
    # cell_nr = [['roi_1', 'roi_6'], ['roi_8', 'roi_22']]
    # fr_rec = 2.0345147125756804

    # onsets = []
    time_axis = []
    for kk, selected_rec in enumerate(f_recs):
        pp = pd.read_csv(f'{save_path}/{selected_rec}_protocol.csv', index_col=0)
        onset = pp['Onset_Time'].iloc[0]
        onset_diff = onset - onset_template
        data_frame = data[data_set[kk]]
        idx = data_frame['rec'] == selected_rec
        selected_data = data_frame[idx].copy()
        col_names = [f'{selected_rec}_{cell_nr[kk][0]}', f'{selected_rec}_{cell_nr[kk][1]}']
        collect_traces = pd.DataFrame(None, columns=col_names)
        # Time axis
        for ii in range(2):
            # selected_data[selected_data['id'] == f'{f_recs[kk]}_{cell_nr[kk][ii]}']
            ca_trace = selected_data[
                (selected_data['anatomy'] == f_labels[kk])
                & (selected_data['id'] == f'{f_recs[kk]}_{cell_nr[kk][ii]}')]['df'].reset_index(drop=True)
            if ii == 0:
                time_axis = uf.convert_samples_to_time(sig=ca_trace, fr=fr_rec)
                # align time axis to template
                time_axis = time_axis - onset_diff
            collect_traces[col_names[ii]] = ca_trace
        # Store example traces to HDD
        collect_traces['Time'] = time_axis
        collect_traces.to_csv(f'{save_path}/example_traces_{f_labels[kk]}.csv')
    uf.msg_box('INFO', 'EXAMPLE TRACES STORED TO HDD (csv files)', '-')


def get_tuning_curve_data(data, save_path):
    # idx1 = data['stimulus_onset_type'] == 'Ramp'
    # ramp_params = data[idx1]['stimulus_onset_parameter'].unique()
    # idx2 = data['stimulus_onset_type'] == 'Step'
    # step_params = data[idx2]['stimulus_onset_parameter'].unique()

    # Drop/Delete bad cell
    # drop_list = ['180509_3_1_roi_11', '180419_8_1_roi_14', '180417_4_1_roi_18',
    #              '180417_4_1_roi_11', '180417_4_1_roi_12']
    # drop_list = ['180509_3_1_roi_11', '180417_2_1_roi_13', '180419_8_1_roi_14']
    drop_list = ['180509_3_1_roi_11']

    f_tags = [
        {'stimulus_onset_type': '=="Step"', 'stimulus_onset_parameter': '==100', 'anatomy': '=="tg"'},
        {'stimulus_onset_type': '=="Step"', 'stimulus_onset_parameter': '==400', 'anatomy': '=="tg"'},
        {'stimulus_onset_type': '=="Step"', 'stimulus_onset_parameter': '==1600', 'anatomy': '=="tg"'},
        {'stimulus_onset_type': '=="Ramp"', 'stimulus_onset_parameter': '==100', 'anatomy': '=="tg"'},
        {'stimulus_onset_type': '=="Ramp"', 'stimulus_onset_parameter': '==200', 'anatomy': '=="tg"'},
        {'stimulus_onset_type': '=="Ramp"', 'stimulus_onset_parameter': '==400', 'anatomy': '=="tg"'},
        {'stimulus_onset_type': '=="Ramp"', 'stimulus_onset_parameter': '==800', 'anatomy': '=="tg"'},
        {'stimulus_onset_type': '=="Ramp"', 'stimulus_onset_parameter': '==1600', 'anatomy': '=="tg"'},
        {'stimulus_onset_type': '=="Step"', 'stimulus_onset_parameter': '==100', 'anatomy': '=="allg"'},
        {'stimulus_onset_type': '=="Ramp"', 'stimulus_onset_parameter': '==100', 'anatomy': '=="allg"'},
        {'stimulus_onset_type': '=="Ramp"', 'stimulus_onset_parameter': '==200', 'anatomy': '=="allg"'},
        {'stimulus_onset_type': '=="Ramp"', 'stimulus_onset_parameter': '==400', 'anatomy': '=="allg"'},
        {'stimulus_onset_type': '=="Ramp"', 'stimulus_onset_parameter': '==800', 'anatomy': '=="allg"'},
        {'stimulus_onset_type': '=="Ramp"', 'stimulus_onset_parameter': '==1600', 'anatomy': '=="allg"'}
    ]
    f_labels = ['01_tg_step_100', '02_tg_step_400', '03_th_step_1600', '04_tg_ramp_100', '05_tg_ramp_200',
                '06_tg_ramp_400', '07_tg_ramp_800', '08_tg_ramp_1600', '01_allg_step_100', '02_allg_ramp_100',
                '03_allg_ramp_200', '04_allg_ramp_400', '05_allg_ramp_800', '06_allg_ramp_1600']

    for kk, vv in enumerate(f_tags):
        f_cell_trials, f_cell_means = get_trials_for_cells(f_data=data, f_tags=f_tags[kk])
        f_cell_means = pd.DataFrame(f_cell_means)
        if drop_list:
            for col_name in drop_list:
                if col_name in f_cell_means.keys():
                    print(f'Ignored: {col_name} in {f_labels[kk]}')
                    f_cell_means = f_cell_means.drop(col_name, axis=1)
        # Store to HDD
        with open(f'{save_path}/tuning_{f_labels[kk]}_all_trials.pkl', 'wb') as f:
            pickle.dump(f_cell_trials, f)
        # with open('saved_dictionary.pkl', 'rb') as f:
        #     loaded_dict = pickle.load(f)
        f_cell_means.to_csv(f'{save_path}/tuning_{f_labels[kk]}.csv', index=False)


def find_tg_sub_types(data, save_path, sub_th=0.5):
    f_tag = {'stimulus_onset_type': '=="Ramp"', 'anatomy': '=="tg"'}
    f_cell_trials, f_cell_means = get_trials_for_cells(f_data=data, f_tags=f_tag)
    f_cell_means = pd.DataFrame(f_cell_means)
    max_response = f_cell_means.iloc[10:20, :].mean()
    idx_sub = max_response >= sub_th
    idx_sub.to_csv(f'{save_path}/subtypes.csv')


def get_data_water_flow():
    rec_name = '221008_11_01'
    rec_dir = f'E:/CaImagingAnalysis/Paper_Data/WaterFlow/{rec_name}/'
    f_raw = pd.read_csv(f'{rec_dir}{rec_name}_raw.txt')
    stimulus = pd.read_csv(f'{rec_dir}{rec_name}_stimulation.txt')
    fr_rec = uf.estimate_sampling_rate(f_raw, stimulus, print_msg=True)
    fr_stimulus = 10000
    stimulus_settings = pd.read_csv(f'{rec_dir}{rec_name}_protocol_stimulation.csv', index_col=0)
    protocol = pd.read_csv(f'{rec_dir}{rec_name}_protocol.csv', index_col=0)
    embed()
    exit()

# Select Data File
# file_dir = uf.select_file([('CSV Files', '.csv')])
# file_dir = 'E:/CaImagingAnalysis/Paper_Data/NilsWenke/recordings/data_frame_complete.csv'
file_dir = 'E:/CaImagingAnalysis/Paper_Data/Sound/Habituation/data_frame_complete.csv'
save_dir = 'E:/CaImagingAnalysis/Paper_Data/Figures/fig_4/data'

get_data_water_flow()
exit()
# df = pd.read_csv(file_dir, index_col=0, low_memory=False).reset_index(drop=True)

# df_audio_cells = pd.read_csv('E:/CaImagingAnalysis/Paper_Data/NilsWenke/TappingAuditoryCells/data_frame_complete.csv',
#                              index_col=0).reset_index(drop=True)

before = 5
after = 25
th_score = 0.1

# sound_get_data_for_matrix_plots(df, save_path=save_dir, f_all_windows=[[1, 15], [1, 4], [5, 200]])
# get_selected_traces(df, save_dir, protocol_template=pd.read_csv(f'{save_dir}/220525_04_01_protocol.csv', index_col=0))
exit()

#
# # --------------------------------------------------------------------------------------------------------------------
# # Get Tuning Curve Data
# find_tg_sub_types(data=df, save_path='E:/CaImagingAnalysis/Paper_Data/Figures/fig_3/data', sub_th=0.5)
# exit()
# get_tuning_curve_data(data=df, save_path='E:/CaImagingAnalysis/Paper_Data/Figures/fig_3/data')
#
# exit()
# # Get Data for Example Single Traces
# get_example_traces(data=[df, df_audio_cells],
#                    save_path='E:/CaImagingAnalysis/Paper_Data/Figures/fig_2/data')
# cell_names = df['rec'].unique()
# c = []
# for k in cell_names:
#     # c.append((df['rec'] == k).sum())
#     idx = df['rec'] == k
#     a = (df[idx]['anatomy'] == 'tg') & (df[idx]['mean_score'] > 0.3)
#     c.append(a.sum())
#
# c = np.array(c)
# max_id = np.where(c > 10)[0]
# most_common_cells = cell_names[max_id]
#
# df = df_audio_cells
# selected_rec = '180418_2_1'
# idx = df['rec'] == selected_rec
# selected_data = df[idx].copy()
# cell_ids = selected_data[selected_data['anatomy'] == 'pllg']['id'].unique()
#
# for k in cell_ids:
#     test = selected_data[(selected_data['anatomy'] == 'pllg') & (selected_data['id'] == k)]['df'].reset_index(drop=True)
#     plt.plot(test, label=k)
#     plt.legend()
#     plt.show()
# embed()
# exit()
# # Store Settings to HDD
# pd.Series(
#     {'before': before, 'after': after, 'th_score': th_score}
# ).to_csv('E:/CaImagingAnalysis/Paper_Data/Figures/fig_2/data/settings.csv')
#
# # Get Data for Matrix Plots (Step/Ramp for TG and ALLG cells)
# get_data_for_matrix_plots(data=df, save_path='E:/CaImagingAnalysis/Paper_Data/Figures/fig_2/data')
# get_audio_cells_for_matrix_plots(data=df_audio_cells, save_path='E:/CaImagingAnalysis/Paper_Data/Figures/fig_2/data')
#
# exit()
#
#
# # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# grand_mean_scores = get_cell_grand_average_score(df)
# idx = grand_mean_scores > th_score
# print(f'score threshold = {th_score}: {grand_mean_scores[idx].shape[0]} / {grand_mean_scores.shape[0]}')
#
#
# # # Estimate tau values for cirf
# # tags = {'anatomy': '=="allg"', 'score': f'>={th_score}'}
# # idx_bool, idx_index = query_data_frame_to_get_index(df, tags)
# # responses, fr = cut_out_windows(df, idx_index, f_before_secs=before, f_after_secs=after, selected_col='df')
# # estimate_tau_values(f_data=responses, f_tags=tags)
#
# # Get cell responses
# # tags = {'stimulus_onset_type': '=="Ramp"', 'anatomy': '=="tg"', 'mean_score': f'>={th_score}'}
# tags = {'stimulus_onset_type': '=="Step"', 'anatomy': '=="pllg"'}
#
# cell_trials, cell_means = get_trials_for_cells(f_data=df_audio_cells, f_tags=tags)
# cell_means = pd.DataFrame(cell_means)
# # sort from min to max responses
# sort_means = cell_means.max(axis=0).sort_values()
# cell_means_sorted = cell_means[sort_means.keys()]
#
# # Create Axis for Matrix Plot
# x = np.arange(0, cell_means.shape[0], 1)
# y = np.arange(1, cell_means.shape[1]+1, 1)
# # Matrix Plot
# plt.figure()
# plt.title(f'cells: {len(y)}, {tags["stimulus_onset_type"][2:]}')
# plt.pcolormesh(x, y, cell_means_sorted.to_numpy().T)
#
# plt.show()
#
#
# exit()
#
# # Set Tags
# tags = {'stimulus_onset_type': '=="Ramp"', 'anatomy': '=="allg"'}
# plot_response(f_data=df, f_tags=tags, f_window=[before, after], activity_measure='z-score')
#
# tags = {'stimulus_onset_type': '=="Step"', 'anatomy': '=="allg"'}
# plot_response(f_data=df, f_tags=tags, f_window=[before, after], activity_measure='z-score')
# plt.show()
#
# exit()
# idx_bool, idx_index = query_data_frame_to_get_index(df, tags)
#
# # Cut out responses
# responses, fr = cut_out_windows(df, idx_index, f_before_secs=3, f_after_secs=20, selected_col='z-score')
# cell_names = df.loc[idx_bool]['id'].unique()
#
# # Compute Mean and STD
# # responses = np.array(responses)
# m = np.mean(responses, axis=0)
# sem = np.std(responses, axis=0) / np.sqrt(len(m))
# t_axis = uf.convert_samples_to_time(sig=m, fr=fr) - 3
#
# # for kk in responses:
# #     plt.plot(kk)
# #     plt.show()
#
# plt.figure()
# plt.title(f'n={cell_names.shape[0]}')
# plt.plot(t_axis, m, 'k')
# plt.plot(t_axis, m-sem, 'r')
# plt.plot(t_axis, m+sem, 'r')
# plt.show()
#
# embed()
# exit()
#
# time_before = 5
# time_after = 20
#
# idx = (df['stimulus_onset_type'] == 'Step') \
#       & (df['stimulus_parameter'] == 100) \
#       & (df['anatomy'] == 'tg') \
#       & (df['mean_score'] > 0.2)
#
# idx_ramp = np.where(idx == True)[0]
# cell_names = df.loc[idx]['id'].unique()
# z_scores = [[]] * len(idx_ramp)
# fr = df.iloc[idx_ramp[0]]['fr']
# for k, s in enumerate(idx_ramp):
#     before = int(time_before * fr)
#     after = int(time_after * fr)
#     z = df.iloc[s-before:s+after]['z-score']
#     z_scores[k] = z
#
# plt.figure()
# plt.title(f'n={cell_names.shape[0]}')
# m = np.mean(z_scores, axis=0)
# sem = np.std(z_scores, axis=0) / np.sqrt(len(m))
# t_axis = uf.convert_samples_to_time(sig=m, fr=fr)
# plt.plot(t_axis, m, 'k')
# plt.plot(t_axis, m-sem, 'r')
# plt.plot(t_axis, m+sem, 'r')
# plt.show()