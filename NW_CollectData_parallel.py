import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import analysis_util_functions as uf
from IPython import embed
import os
import time
from joblib import Parallel, delayed


def import_f_raw(file_dir):
    data = pd.read_csv(file_dir, decimal='.', sep='\t', header=None).drop(columns=0)
    header_labels = []
    for kk in range(data.shape[1]):
        header_labels.append(f'roi_{kk + 1}')
    data.columns = header_labels
    return data


def collect_data(rec_dir):
    output_data_frame = pd.DataFrame()
    # anatomy = dict.fromkeys(rec_list)
    rec_name = os.path.split(rec_dir)[1]
    rec_dob = os.path.split(os.path.split(rec_dir)[0])[1][:-4]
    # anatomy[rec_name] = pd.read_csv(f'{rec_dir}/{rec_name}_anatomy.csv')
    anatomy = pd.read_csv(f'{rec_dir}/{rec_name}_selected_cells.csv', index_col=0)
    # Ignore this recording if there are no good cells (no entries in anatomy.csv)
    if anatomy.shape[0] > 0:
        # get raw values of this recording
        # f_raw = import_f_raw(f'{rec_dir}/{rec_name}_raw.txt')
        f_raw = pd.read_csv(f'{rec_dir}/{rec_name}_raw.txt')
        # Get only the good cells (based on LM score)
        # f_raw_selected = f_raw[anatomy[rec_name].keys()]
        try:
            # f_raw_selected = f_raw[anatomy[rec_name]['roi'].transpose().to_numpy()]
            f_raw_selected = f_raw[anatomy['roi'].transpose().to_numpy()]
        except:
            print(f'ERROR in: {rec_name}')
            print(f_raw[f_raw.keys()[1]])

        # Filter Raw Data
        f_raw_selected_filtered = uf.filter_raw_data(sig=f_raw_selected, win=7, o=3)

        # Compute f base line (5th percentile of all values)
        fbs = np.percentile(f_raw_selected_filtered, 5, axis=0)

        # Compute delta f over f
        df_f = (f_raw_selected_filtered - fbs) / fbs

        # Compute z-scores
        z_scores = (df_f - np.mean(df_f, axis=0)) / np.std(df_f, axis=0)

        # get stimulus
        # stimulus = uf.import_txt_stimulation_file(f'{rec_dir}/', f'{rec_name}_stimulation', float_dec='.')
        stimulus = pd.read_csv(f'{rec_dir}/{rec_name}_stimulation.txt')
        fr_rec = uf.estimate_sampling_rate(f_raw_selected, f_stimulation=stimulus, print_msg=False)
        time_axis = uf.convert_samples_to_time(f_raw_selected, fr=fr_rec)

        # get protocol
        protocol = pd.read_csv(f'{rec_dir}/{rec_name}_protocol.csv', index_col=0)

        # get linear regression model results
        lm = np.load(f'{rec_dir}/{rec_name}_lm_results.npy', allow_pickle=True).item()

        # Compute Calcium Impulse Response Function (CIRF)
        cirf_tau = 10
        cirf = uf.create_cif(fr_rec, tau=cirf_tau)

        # Compute Regressor for the entire stimulus trace
        binary, reg, _, _ = uf.create_binary_trace(
            sig=f_raw, cirf=cirf, start=protocol['Onset_Time'], end=protocol['Offset_Time'],
            fr=fr_rec, ca_delay=0, pad_before=5, pad_after=20, low=0, high=1
        )
        # Normalize Regressor to max = 1
        reg_norm = reg / np.max(reg)
        # Correct for too long Regressors (due to convolution ...)
        reg_norm = reg_norm[:len(binary)]

        # Get List of used stimuli
        step_parameters = protocol[protocol['Stimulus'] == 'Step']['Duration'].unique()
        ramp_parameters = protocol[protocol['Stimulus'] == 'Ramp']['Duration'].unique()
        stimulus_used = pd.DataFrame()
        stimulus_used['parameter'] = np.append(step_parameters, ramp_parameters)
        stimulus_used['type'] = np.append(['Step'] * len(step_parameters), ['Ramp'] * len(ramp_parameters))
        stimulus_used['count'] = [0] * len(stimulus_used)

        # stimulus_onsets_type = ['NaN'] * len(f_raw_selected)
        stimulus_onsets_type = pd.DataFrame(np.zeros((len(f_raw_selected))) * np.nan)
        stimulus_onsets_parameter = stimulus_onsets_type.copy()
        stimulus_offsets_type = stimulus_onsets_type.copy()
        stimulus_offsets_parameter = stimulus_onsets_type.copy()
        stimulus_trial = stimulus_onsets_type.copy()
        # stimulus_trial = pd.DataFrame(np.zeros((len(f_raw_selected))) * np.nan)

        trial_scores = pd.DataFrame(
            np.zeros((len(f_raw_selected), len(f_raw_selected.keys()))),
            columns=f_raw_selected.keys()
        )
        mean_scores_rois = trial_scores.copy()

        for kk in range(protocol.shape[0]):
            onset = int(protocol['Onset_Time'].iloc[kk] * fr_rec)
            offset = int(protocol['Offset_Time'].iloc[kk] * fr_rec)

            s_type = protocol['Stimulus'].iloc[kk]
            s_parameter = protocol['Duration'].iloc[kk]

            # Set trial
            idx_trial = (stimulus_used['type'] == s_type) & (stimulus_used['parameter'] == s_parameter)
            stimulus_used.loc[idx_trial, 'count'] = stimulus_used.loc[idx_trial, 'count'] + 1

            # Mark exact onset and offset point
            stimulus_onsets_type.iloc[onset] = s_type
            stimulus_onsets_parameter.iloc[onset] = s_parameter
            stimulus_offsets_type.iloc[offset] = s_type
            stimulus_offsets_parameter.iloc[offset] = s_parameter

            # Trial Number on Stimulus onset time
            trial = stimulus_used.loc[idx_trial, 'count'].item()
            stimulus_trial.iloc[onset] = trial

            s_name = f'{s_type}-{s_parameter}'
            for key in f_raw_selected:
                trial_scores[key].iloc[onset] = np.round(lm[key][s_name]['score'][trial-1], 5)
                mean_scores_rois[key].iloc[onset] = np.round(np.mean(lm[key][s_name]['score'], axis=0), 5)

        # Now loop through all rois
        for kk, roi in enumerate(f_raw_selected):
            size = len(f_raw_selected)
            roi_data = pd.DataFrame()
            roi_data['id'] = [f'{rec_name}_{roi}'] * size
            roi_data['rec'] = [f'{rec_name}'] * size
            roi_data['roi'] = [roi] * size
            roi_data['raw'] = f_raw_selected[roi]
            roi_data['raw_filtered'] = f_raw_selected_filtered[roi]
            roi_data['df'] = df_f[roi]
            roi_data['z-score'] = z_scores[roi]
            roi_data['fbs'] = fbs[kk]
            roi_data['cirf_tau'] = cirf_tau
            # roi_data['anatomy'] = [anatomy[rec_name][roi].item()] * size
            idx_anatomy = anatomy['roi'] == roi
            roi_data['anatomy'] = [anatomy['anatomy'][idx_anatomy].item()] * size
            roi_data['time'] = time_axis
            roi_data['stimulus_onset_type'] = stimulus_onsets_type
            roi_data['stimulus_onset_parameter'] = stimulus_onsets_parameter
            roi_data['stimulus_offset_type'] = stimulus_offsets_type
            roi_data['stimulus_offset_parameter'] = stimulus_offsets_parameter
            roi_data['trial'] = stimulus_trial
            roi_data['mean_score'] = mean_scores_rois[roi]
            roi_data['score'] = trial_scores[roi]
            roi_data['dob'] = rec_dob
            roi_data['rec_date'] = rec_name[:6]
            roi_data['binary'] = binary
            roi_data['regressor'] = reg_norm
            roi_data['fr'] = fr_rec
            roi_data['sample'] = np.linspace(1, size, size).astype('int')
            output_data_frame = pd.concat([output_data_frame, roi_data], ignore_index=True)
        print(f'--- {rec_name} ---')
        return [output_data_frame, rec_name]
    return None


if __name__ == '__main__':
    base_dir = uf.select_dir()
    # Ignore all files and only take directories
    dob_count = len([s for s in os.listdir(base_dir) if '.' not in s])

    all_dirs = []
    for root, dirs, files in os.walk(base_dir, topdown=True):
        for name in dirs:
            # print(os.path.join(root, name))
            all_dirs.append(os.path.join(root, name))

    rec_list = [s for s in all_dirs if 'refs' not in s]
    rec_list = rec_list[dob_count:]

    uf.msg_box(f'INFO', 'COLLECTING DATA ... PLEASE WAIT ...', '+')
    t0 = time.perf_counter()

    # for k, v in enumerate(rec_list): print(k, v)
    # a = collect_data(rec_list[0])
    # Start Parallel Loop to do all recordings in parallel
    # result will be a list of all outputs
    result = Parallel(n_jobs=-1)(delayed(collect_data)(i) for i in rec_list)
    t1 = time.perf_counter()

    # Combine all recordings into one data frame
    data_frame = pd.DataFrame()
    recording_list = []
    for k, r in enumerate(result):
        if r:
            data_frame = pd.concat([data_frame, r[0]])
            recording_list.append(r[1])

    uf.msg_box('INFO', f'Collecting Data took: {np.round(t1 - t0, 2)} secs \n'
                       f'Data Frame Size: {len(data_frame):,}', '+')

    # Store Data Frame as csv file to HDD
    data_frame.to_csv(f'{base_dir}/data_frame_complete.csv')
    recording_list = pd.DataFrame(recording_list, columns=['Recording'])
    recording_list.to_csv(f'{base_dir}/recording_list.csv', columns=['Recording'])

    t2 = time.perf_counter()
    uf.msg_box('INFO', f'Storing Data Frame to HDD took: {np.round(t2-t1, 2)} secs', '+')
