import numpy as np
import pandas as pd
import analysis_util_functions as uf
from IPython import embed
from read_roi import read_roi_zip
import os
from joblib import Parallel, delayed
import time

""" This Script will perform a Linear Regression Model between Cell Responses (Ca-Imaging) and Stimulus Trace to get
    a score for each cell in the recording 
"""


def linear_regression_scoring(file_dir, protocol_style):
    # SETTINGS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    score_th = 0.1
    cirf_tau = 3  # in secs
    fbs_percentile = 5  # in %
    fbs_window = 100  # in secs
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    data_file_name = os.path.split(file_dir)[1]
    # rec_dir = os.path.split(file_dir)[0]
    rec_dir = file_dir
    # rec_name = os.path.split(rec_dir)[1]
    rec_name = os.path.split(file_dir)[1]
    data_file_name = f'{rec_name}_raw.txt'

    print(f'-- {rec_name} --')

    # Import stimulation trace
    stimulus = uf.import_txt_stimulation_file(f'{rec_dir}', f'{rec_name}_stimulation', float_dec='.')

    # Import Protocol
    protocol = pd.read_csv(f'{rec_dir}/{rec_name}_protocol.csv', index_col=0)

    # Import raw fluorescence traces (rois)
    # It is important that the header is equal to the correct ROI number
    # All ROIS of the recording
    f_raw = uf.import_f_raw(f'{rec_dir}/{data_file_name}')

    # Estimate frame rate
    # fr_rec = uf.estimate_sampling_rate(data=f_raw, f_stimulation=stimulus, print_msg=False)
    fr_rec = 2.0345147125756804
    # padding settings
    pad_before = 5  # in secs
    pad_after = 20  # in secs

    if protocol_style == 'sound':
        # Get step and ramp stimuli
        single_parameters = protocol[protocol['Stimulus'] == 'SinglePulse']['Duration'].unique()
        train_parameters = protocol[protocol['Stimulus'] == 'TrainPulse']['Duration'].unique()
        stimulus_parameters = pd.DataFrame()
        stimulus_parameters['parameter'] = np.append(single_parameters, train_parameters)
        stimulus_parameters['type'] = np.append(['SinglePulse'] * len(single_parameters), ['TrainPulse'] * len(train_parameters))
    else:
        # Get step and ramp stimuli
        step_parameters = protocol[protocol['Stimulus'] == 'Step']['Duration'].unique()
        ramp_parameters = protocol[protocol['Stimulus'] == 'Ramp']['Duration'].unique()

        stimulus_parameters = pd.DataFrame()
        stimulus_parameters['parameter'] = np.append(step_parameters, ramp_parameters)
        stimulus_parameters['type'] = np.append(['Step'] * len(step_parameters), ['Ramp'] * len(ramp_parameters))

    # Compute delta f over f
    # f_raw_filtered = uf.filter_raw_data(sig=f_raw, win=filter_window, o=filter_order)
    # fbs = np.percentile(f_raw, fbs_percentile, axis=0)
    # df_f = (f_raw-fbs) / fbs
    # NO Filtering, use sliding window of 100 s for computing dynamic fbs
    df_f, _, _ = uf.compute_df_over_f(f_values=f_raw, window_size=fbs_window, per=fbs_percentile, fast=True)

    # Compute Calcium Impulse Response Function (CIRF)
    cirf = uf.create_cif(fr_rec, tau=cirf_tau)

    # Select Stimulus Type
    label = []
    for k in range(stimulus_parameters.shape[0]):
        label.append(f'{stimulus_parameters.iloc[k, 1]}-{stimulus_parameters.iloc[k, 0]}')

    # Create Binary and cut out ROI Ca responses for each stimulus type (and its repetitions)
    stimulus_regression = []
    for k in range(stimulus_parameters.shape[0]):
        stim = protocol[(protocol['Stimulus'] == stimulus_parameters.iloc[k, 1]) &
                        (protocol['Duration'] == stimulus_parameters.iloc[k, 0])]
        binary, reg, response_c, reg_c = uf.create_binary_trace(
            sig=df_f, cirf=cirf, start=stim['Onset_Time'], end=stim['Offset_Time'],
            fr=fr_rec, ca_delay=0, pad_before=pad_before, pad_after=pad_after, low=0, high=1
        )
        dummy = {'binary': binary, 'reg': reg, 'response_cutout': response_c, 'reg_cutout': reg_c}
        stimulus_regression.append(dummy)

    # Linear Regression Model to Score Responses to Stimulus Types
    all_cells = dict.fromkeys(df_f.keys())
    for rois in df_f:  # loop through rois (cells)
        # lm = []
        lm2 = dict.fromkeys(label)
        for k, stim_type in enumerate(stimulus_regression):  # loop through stimulus types (6)
            score, r_squared, slope, response_trace = [], [], [], []
            for kk, repeat_reg in enumerate(stim_type['reg_cutout']):  # loop through repetitions (5)
                dummy_response = stim_type['response_cutout'][kk][rois]
                # Linear Model for one cell, one stimulus type and one repetition
                try:
                    sc, rr, aa = uf.apply_linear_model(xx=repeat_reg, yy=dummy_response, norm_reg=True)
                except ValueError:
                    print('Could not compute linear model for this trial')
                    sc, rr, aa = 0, 0, 0
                score.append(sc)
                r_squared.append(rr)
                slope.append(aa)
                response_trace.append(dummy_response)
            # lm.append({'score': score, 'r_squared': r_squared, 'slope': slope, 'response': response_trace})
            lm2[label[k]] = {'score': score, 'r_squared': r_squared, 'slope': slope, 'response': response_trace}
            # all stimulus types for one cell
        all_cells[rois] = lm2
    # Compute Mean Score Values
    mean_score = dict.fromkeys(df_f)
    mean_r_squared = dict.fromkeys(df_f)
    mean_slope = dict.fromkeys(df_f)
    for rois in df_f:
        score_stim_type_means = []
        r_squared_stim_type_means = []
        slope_stim_type_means = []
        response_means = []
        for k, stim_type in enumerate(stimulus_regression):
            score_stim_type_means.append(np.mean(all_cells[rois][label[k]]['score']))
            r_squared_stim_type_means.append(np.mean(all_cells[rois][label[k]]['r_squared']))
            slope_stim_type_means.append(np.mean(all_cells[rois][label[k]]['slope']))
            response_means.append(np.mean(all_cells[rois][label[k]]['response']))

        mean_score[rois] = score_stim_type_means
        mean_r_squared[rois] = r_squared_stim_type_means
        mean_slope[rois] = slope_stim_type_means

    # Indices are the different stimulus types (6) showing mean score values (all 5 repetitions)
    mean_score = pd.DataFrame(mean_score)
    final_mean_score = pd.concat([stimulus_parameters, mean_score], axis=1)

    # Sort out cells by LM score
    good_cells_by_score = []
    for key in final_mean_score.keys()[2:]:
        if any(final_mean_score[key] >= score_th):
            good_cells_by_score.append(key)

    # Store Linear Regression Results to HDD
    # load it with np.load(dir, allow_pickle).item()
    good_cells_by_score_csv = pd.DataFrame(good_cells_by_score, columns=['roi'])
    good_cells_by_score_csv.to_csv(f'{rec_dir}/{rec_name}_lm_good_score_rois.csv')
    np.save(f'{rec_dir}/{rec_name}_lm_results.npy', all_cells)
    final_mean_score.to_csv(f'{rec_dir}/{rec_name}_lm_mean_scores.csv')
    # Create Anatomy Labels
    # # check if there is already a anatomy label
    # check = [s for s in os.listdir(rec_dir) if 'anatomy' in s]
    # if not check:
    #     anatomy = pd.DataFrame(columns=good_cells_by_score_csv['roi'])
    #     anatomy.to_csv(f'{rec_dir}/{rec_name}_anatomy.csv')


if __name__ == '__main__':
    # Select Directory containing several recordings
    base_dir = uf.select_dir()

    t0 = time.perf_counter()
    uf.msg_box('INFO', f'Linear Regression Scoring is starting ...', '+')
    linear_regression_scoring(base_dir, protocol_style='sound')
    t1 = time.perf_counter()
    uf.msg_box('INFO', f'Linear Regression Scoring finished!\n'
                       f'It took: {np.round(t1-t0, 2)} s', '+')
