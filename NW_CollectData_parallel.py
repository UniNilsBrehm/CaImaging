import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import analysis_util_functions as uf
from IPython import embed
import os
import time
from joblib import Parallel, delayed


def import_f_raw(file_dir):
    data = pd.read_csv(file_dir, decimal='.', sep='\t', header=None)
    header_labels = []
    for kk in range(data.shape[1]):
        header_labels.append(f'roi_{kk + 1}')
    data.columns = header_labels
    return data


def collect_data(rec_dir):
    output_data_frame = pd.DataFrame()
    anatomy = dict.fromkeys(rec_list)
    rec_name = os.path.split(rec_dir)[1]
    anatomy[rec_name] = pd.read_csv(f'{rec_dir}/{rec_name}_anatomy.csv')
    # Ignore this recording if there are no good cells (no entries in anatomy.csv)
    if anatomy[rec_name].shape[0] > 0:
        # get raw values of this recording
        f_raw = import_f_raw(f'{rec_dir}/{rec_name}_raw.txt')
        # Get only the good cells (based on LM score)
        f_raw_selected = f_raw[anatomy[rec_name].keys()]

        # Filter Raw Data
        f_raw_selected_filtered = uf.filter_raw_data(sig=f_raw_selected, win=7, o=3)

        # Compute f base line (5th percentile of all values)
        fbs = np.percentile(f_raw_selected_filtered, 5, axis=0)

        # Compute delta f over f
        df_f = (f_raw_selected_filtered - fbs) / fbs

        # Compute z-scores
        z_scores = (df_f - np.mean(df_f, axis=0)) / np.std(df_f, axis=0)

        # get stimulus
        stimulus = uf.import_txt_stimulation_file(f'{rec_dir}/', f'{rec_name}_stimulation', float_dec='.')
        fr_rec = uf.estimate_sampling_rate(f_raw_selected, f_stimulation=stimulus, print_msg=False)
        time_axis = uf.convert_samples_to_time(f_raw_selected, fr=fr_rec)

        # get protocol
        protocol = pd.read_csv(f'{rec_dir}/{rec_name}_protocol.csv')

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
        stimulus_onsets_type = ['NaN'] * len(f_raw_selected)
        stimulus_onsets_parameter = ['NaN'] * len(f_raw_selected)
        stimulus_offsets_type = ['NaN'] * len(f_raw_selected)
        stimulus_offsets_parameter = ['NaN'] * len(f_raw_selected)
        stimulus_trial = ['NaN'] * len(f_raw_selected)
        mean_scores_rois = dict.fromkeys(f_raw_selected, ['NaN'] * len(f_raw_selected))
        trial_scores = dict.fromkeys(f_raw_selected, ['NaN'] * len(f_raw_selected))
        for k in range(protocol.shape[0]):
            onset = int(protocol['Onset_Time'].iloc[k] * fr_rec)
            offset = int(protocol['Offset_Time'].iloc[k] * fr_rec)

            s_type = protocol['Stimulus'].iloc[k]
            s_parameter = protocol['Duration'].iloc[k]

            # Set trial
            idx_trial = (stimulus_used['type'] == s_type) & (stimulus_used['parameter'] == s_parameter)
            stimulus_used.loc[idx_trial, 'count'] = stimulus_used.loc[idx_trial, 'count'] + 1

            # Mark exact onset and offset point
            stimulus_onsets_type[onset] = s_type
            stimulus_onsets_parameter[onset] = s_parameter

            stimulus_offsets_type[offset] = s_type
            stimulus_offsets_parameter[offset] = s_parameter

            # Trial Number on Stimulus onset time
            trial = stimulus_used.loc[idx_trial, 'count'].item()
            stimulus_trial[onset] = trial

            s_name = f'{s_type}-{s_parameter}'
            for key in f_raw_selected:
                trial_scores[key][onset] = np.round(lm[key][s_name]['score'][trial-1], 4)
                mean_scores_rois[key][onset] = np.round(np.mean(lm[key][s_name]['score']), 4)

        # Now loop through all rois
        for k, roi in enumerate(f_raw_selected):
            size = len(f_raw_selected)
            roi_data = pd.DataFrame()
            roi_data['id'] = [f'{rec_name}_{roi}'] * size
            roi_data['rec'] = [f'{rec_name}'] * size
            roi_data['roi'] = [roi] * size
            roi_data['raw'] = f_raw_selected[roi]
            roi_data['raw_filtered'] = f_raw_selected_filtered[roi]
            roi_data['df'] = df_f[roi]
            roi_data['z-score'] = z_scores[roi]
            roi_data['fbs'] = fbs[k]
            roi_data['cirf_tau'] = cirf_tau
            roi_data['anatomy'] = [anatomy[rec_name][roi].item()] * size
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

            output_data_frame = pd.concat([output_data_frame, roi_data])
        return output_data_frame


if __name__ == '__main__':
    base_dir = uf.select_dir()
    rec_dob = os.path.split(base_dir)[1][:-4]
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

    # Start Parallel Loop to do all recordings in parallel
    result = Parallel(n_jobs=-1)(delayed(collect_data)(i) for i in rec_list)
    t1 = time.perf_counter()

    # Combine all recordings into one data frame
    data_frame = pd.DataFrame()
    for r in result:
        data_frame = pd.concat([data_frame, r])

    uf.msg_box('INFO', f'Collecting Data took: {np.round(t1 - t0, 2)} secs \n'
                       f'Data Frame Size: {len(data_frame):,}', '+')

    # Store Data Frame as csv file to HDD
    data_frame.to_csv(f'{base_dir}/data_frame_complete.csv')
    t2 = time.perf_counter()
    uf.msg_box('INFO', f'Storing Data Frame to HDD took: {np.round(t2-t1, 2)} secs', '+')
