import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter.filedialog import askdirectory
from tkinter.filedialog import askopenfilename
from tkinter import Tk
from IPython import embed
import analysis_util_functions as uf
import csv
import sys


def import_stimulation_file(file_dir):
    # Check for header and delimiter
    with open(file_dir) as csv_file:
        some_lines = csv_file.read(1024)
        dialect = csv.Sniffer().sniff(some_lines)
        delimiter = dialect.delimiter

    # Load data and assume the first row is the header
    data = pd.read_csv(file_dir, decimal='.', sep=delimiter, header=0, index_col=0)
    # Chek for correct header
    embed()
    exit()
    try:
        a = float(data.keys()[0])  # this means no headers in the file
        data = pd.read_csv(file_dir, decimal='.', sep=delimiter, header=None, index_col=0)
        data.columns = ['Time', 'Volt']
        data = data.reset_index(drop=True)
        return data
    except ValueError:
        data = data.drop(data.columns[0], axis=1)
        return data


def import_nw_stimulation_file(s_file_path):
    df = pd.read_csv(s_file_path, sep='\t')
    return df


def convert_nw_stimulation_file(s_file):
    s_b_idx = [0, 1000]
    # take the last col (= to col size-1)
    idx = s_file.shape[1]-1
    s_stimulus, _ = interpolate_stimulus(s_file.iloc[:, idx].copy().to_numpy(), s_b=s_b_idx)
    return s_stimulus


def correct_stimulus_traces(s1):
    # Find High Freq Noise
    idx1 = np.where(np.diff(s1) > 0.5)
    s1[idx1] = np.NaN
    # Set Base Line to Zero
    s1 = s1 - np.nanmean(s1)
    return s1, idx1


def interpolate_stimulus(s2, s_b):
    # Find High Freq Noise
    idx2 = np.where(np.diff(s2) > 0.5)
    for k in idx2:
        s2[k] = s2[k+1]
    # Set Base Line to Zero
    s2 = s2 - np.nanmean(s2[s_b[0]:s_b[1]])
    return s2, idx2


def find_stimulus_time(volt_threshold, f_stimulation, mode):
    # Find stimulus time points
    if mode == 'below':
        threshold_crossings = np.diff(f_stimulation < volt_threshold, prepend=False)
    else:
        mode = 'above'
        threshold_crossings = np.diff(f_stimulation > volt_threshold, prepend=False)

    # Get Upward Crossings
    f_upward = np.argwhere(threshold_crossings)[::2, 0]  # Upward crossings

    # Get Downward Crossings
    f_downward = np.argwhere(threshold_crossings)[1::2, 0]  # Downward crossings

    return f_downward, f_upward


def interval_thresholding(x, x_th):
    # Look for gap duration
    # Then threshold this gap duration
    # if too small throw away the second of the pair
    diffs = np.diff(x)
    idx = diffs > x_th
    idx = np.insert(idx, 0, True)
    y = x[idx]
    return y, idx


def find_single_steps_and_trains(s, f_th_step, interval_th):
    # Find steps with diff
    f_stimulus_diff = np.diff(s, append=0)
    _, f_step_onsets = find_stimulus_time(volt_threshold=f_th_step, f_stimulation=f_stimulus_diff, mode='above')
    f_step_offsets, _ = find_stimulus_time(volt_threshold=-f_th_step, f_stimulation=f_stimulus_diff, mode='below')
    # Find Single Pulses and Trains by intervals
    intervals = np.diff(f_step_onsets)
    intervals = np.append(intervals, 20000)
    single_pulses_onsets = []
    single_pulses_offsets = []
    train_pulses_onsets = []
    train_pulses_offsets = []
    for ii, vv in enumerate(f_step_onsets):
        # First pulse in stimulus trace:
        if ii == 0:
            if intervals[0] >= interval_th:
                single_pulses_onsets.append(vv)
                single_pulses_offsets.append(f_step_offsets[ii])
            else:
                train_pulses_onsets.append(vv)
                train_pulses_offsets.append(f_step_offsets[ii])
        else:  # Following pulses in stimulus trace:
            if (intervals[ii] >= interval_th) & (intervals[ii-1] >= interval_th):
                single_pulses_onsets.append(vv)
                single_pulses_offsets.append(f_step_offsets[ii])
            else:
                train_pulses_onsets.append(vv)
                train_pulses_offsets.append(f_step_offsets[ii])

    # Get only trains trace and single steps trace separately
    f_trains = s.copy()
    f_single = np.zeros(len(s))
    for kk in range(len(single_pulses_onsets)):
        f_trains[single_pulses_onsets[kk]:single_pulses_offsets[kk]] = 0
        f_single[single_pulses_onsets[kk]:single_pulses_offsets[kk]] = int(np.round(np.max(s), 1))

    # Find first pulse in the pulse train
    p_trains_intervals = np.diff(train_pulses_onsets)
    idx_first_pulses = p_trains_intervals >= 13000
    idx_first_pulses = np.insert(idx_first_pulses, 0, True)
    train_first_pulses_onsets = np.array(train_pulses_onsets)[idx_first_pulses]
    train_first_pulses_offsets = np.array(train_pulses_offsets)[idx_first_pulses]

    # Delete First Train Pulses from Train Pulses
    train_pulses_onsets = np.array(train_pulses_onsets)[np.invert(idx_first_pulses)]
    train_pulses_offsets = np.array(train_pulses_offsets)[np.invert(idx_first_pulses)]
    return f_single, f_trains, single_pulses_onsets, single_pulses_offsets, train_pulses_onsets, train_pulses_offsets, train_first_pulses_onsets, train_first_pulses_offsets


def find_ramps_and_steps(s, f_th_step, f_th_ramp, interval_th):
    # Find steps with diff
    f_stimulus_diff = np.diff(s, append=0)
    _, f_step_onsets = find_stimulus_time(volt_threshold=f_th_step, f_stimulation=f_stimulus_diff, mode='above')
    f_step_offsets, _ = find_stimulus_time(volt_threshold=-f_th_step, f_stimulation=f_stimulus_diff, mode='below')

    # Remove Steps from stimulus trace
    f_ramp_stimulus = s.copy()
    for kk in range(len(f_step_onsets)):
        f_ramp_stimulus[f_step_onsets[kk]:f_step_offsets[kk]] = 0

    # Now find the Ramps
    f_ramp_offsets, f_ramp_onsets = find_stimulus_time(volt_threshold=f_th_ramp, f_stimulation=f_ramp_stimulus, mode='above')

    # Thresholding for to small intervals
    if interval_th > 0:
        f_ramp_onsets, idx = interval_thresholding(f_ramp_onsets, x_th=interval_th)
        f_ramp_offsets = f_ramp_offsets[idx]

        f_step_onsets, idx = interval_thresholding(f_step_onsets, x_th=interval_th)
        f_step_offsets = f_step_offsets[idx]

    return f_ramp_stimulus, f_step_onsets, f_step_offsets, f_ramp_onsets, f_ramp_offsets


def find_distance_to_max_val(f_stimulus, f_step_onsets, f_step_offsets):
    # Find Distance from Detection Point to Max Point in Samples
    f_samples_to_max = []
    for kk in range(len(f_step_onsets)):
        cc = f_stimulus[f_step_onsets[kk]:f_step_offsets[kk]]
        f_samples_to_max.append(np.where(cc == np.max(cc))[0][0])
    return f_samples_to_max


def compare_stimulus_duration(f_protocol, f_estimated_durations):
    f_durations = []
    for kk in range(len(f_estimated_durations)):
        # Compare each duration in protocol file with the estimated one
        f_diff = abs(f_protocol['Duration'] - f_estimated_durations[kk])
        # Find the minimal difference and match it
        f_m = f_protocol['Duration'][f_diff == f_diff.min()].values[0]
        f_durations.append(f_m)
    return f_durations


def open_dir(f_select_single_file):
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    if f_select_single_file:
        # SELECT STIMULATION FILE (.txt)
        print('-------------------- INFO --------------------')
        print('PLEASE SELECT STIMULATION FILE (.txt)')
        print('-------------------- INFO --------------------')
        rec_path = askopenfilename()
    else:
        # SELECT RECORDING DIRECTORY: which includes 'stimulation_data.txt'
        print('-------------------- INFO --------------------')
        print('PLEASE SELECT RECORDING DIRECTORY')
        print('-------------------- INFO --------------------')
        rec_path = askdirectory()
    return rec_path


def detect_stimuli(s_values, protocol_file=False, th_step=0.5, th_ramp=0.2, small_interval_th=0, smoothing_window=10,
                   show_helper_figure=False, nils_wenke=False, protocol_values=0):

    # Import Stimulus Protocols (hardcoded dir)
    # 6 stimulus types, each repeated for 5 times (total: 30)
    protocol_path = 'E:/CaImagingAnalysis/NilsWenke/'
    if protocol_file:
        if nils_wenke:
            protocol_1 = pd.read_csv(f'{protocol_path}stimulus_protocol_1.csv')
            protocol_2 = pd.read_csv(f'{protocol_path}stimulus_protocol_2.csv')
        else:
            # protocol_values = pd.read_csv('E:/CaImagingAnalysis/Paper_Data/Tapping/protocol_stimulus_values.csv')
            protocol_values = protocol_values
    # Smooth Stimulus
    # stimulus = s_values
    # stimulus = uf.savitzky_golay(y=s_values, window_size=3, order=1)
    stimulus = np.convolve(s_values, np.ones(smoothing_window) / smoothing_window, mode='same')

    # Find Ramps and Steps in Stimulus Trace
    # Thresholding too small intervals
    # Min. distance between different stimuli in samples that is allowed!
    # This somehow corresponds to the stimulus intervals in the experiment (here: 30 secs)
    # This means it will ignore any stimulus detection x secs after the first detection! So the first must be correct!
    # I guess this could be improved... However, if 'th_step' and 'th_ramp' are set appropriately this small interval
    # thresholding should not be necessary anyways.
    ramp_stimulus, step_onsets, step_offsets, ramp_onsets, ramp_offsets = find_ramps_and_steps(
        s=stimulus.copy(), f_th_step=th_step, f_th_ramp=th_ramp, interval_th=small_interval_th
    )

    if show_helper_figure:
        fig, axs = plt.subplots(2, 1)
        axs[0].set_title('Stimulus Diff (red) (Estimate good step threshold value)')
        axs[0].plot(np.diff(stimulus, append=0), 'k')
        axs[0].plot([0, len(stimulus)], [th_step, th_step], 'r--')
        axs[1].plot(stimulus, 'k')
        axs[1].plot(step_onsets, np.zeros(len(step_onsets)) + th_step, 'xb', markersize=10)
        axs[1].plot(step_offsets, np.zeros(len(step_offsets)) + th_step, 'xb', markersize=10)
        axs[1].plot(ramp_onsets, np.zeros(len(ramp_onsets)) + th_ramp, 'xg', markersize=10)
        axs[1].plot(ramp_offsets, np.zeros(len(ramp_offsets)) + th_ramp, 'xg', markersize=10)
        axs[1].plot([0, len(stimulus)], [th_ramp, th_ramp], 'r--')
        axs[1].set_title('Stimulus Trace')
        plt.show()

    if protocol_file:
        if nils_wenke:
            metadata_df = pd.DataFrame()
            # WHAT PROTOCOL HAS BEEN USED?
            if len(step_onsets) > 10:
                protocol = protocol_2.copy()
                metadata_df['Protocol'] = ['Wenke Protocol Nr. 2']
                print('It seems that protocol 2 has been used!')
            else:
                protocol = protocol_1.copy()
                metadata_df['Protocol'] = ['Wenke Protocol Nr. 1']
                print('It seems that protocol 1 has been used!')
        else:
            protocol = protocol_values

    # Determine ramp and step durations
    fr = 1000
    stimulus_time = uf.convert_samples_to_time(stimulus, fr=fr)
    estimated_step_durations = step_offsets - step_onsets
    estimated_ramp_durations = ramp_offsets - ramp_onsets

    # Compare estimated Step duration values to protocol files
    # STEPS
    if protocol_file:
        protocol_steps = protocol[protocol['Stimulus'] == 'Step'].copy()
        step_durations = compare_stimulus_duration(
            f_protocol=protocol_steps.copy(), f_estimated_durations=estimated_step_durations
        )
        # RAMPS
        # Use 'estimated_ramp_durations / 2' to match it to Nils Wenkes Terminology!
        protocol_ramps = protocol[protocol['Stimulus'] == 'Ramp'].copy()
        ramp_matched_durations = compare_stimulus_duration(
            f_protocol=protocol_ramps.copy(), f_estimated_durations=estimated_ramp_durations / 2
        )

    # Combine all info into one df
    # RAMPS
    ramps_df = pd.DataFrame()
    if protocol_file:
        ramps_df['Duration'] = ramp_matched_durations
        ramps_df['Estimated_Duration'] = estimated_ramp_durations / 2
        ramps_df['Onset_Sample'] = ramp_onsets
        ramps_df['Onset_Time'] = ramp_onsets / fr
        ramps_df['Offset_Sample'] = ramp_offsets
        ramps_df['Offset_Time'] = ramp_onsets / fr + np.array(ramp_matched_durations) / fr
        ramps_df['Detected_Offset_Time'] = ramp_offsets / fr
        ramps_df['Stimulus'] = ['Ramp'] * len(ramp_matched_durations)
        ramps_df['SamplingRate'] = [fr] * len(ramp_matched_durations)
    else:
        ramps_df['Estimated_Duration'] = estimated_ramp_durations / 2
        ramps_df['Onset_Sample'] = ramp_onsets
        ramps_df['Onset_Time'] = ramp_onsets / fr
        ramps_df['Offset_Sample'] = ramp_offsets
        ramps_df['Offset_Time'] = ramp_onsets / fr + np.array(estimated_ramp_durations) / fr
        ramps_df['Detected_Offset_Time'] = ramp_offsets / fr
        ramps_df['Stimulus'] = ['Ramp'] * len(estimated_ramp_durations)
        ramps_df['SamplingRate'] = [fr] * len(estimated_ramp_durations)

    # STEPS
    steps_df = pd.DataFrame()
    if protocol_file:
        steps_df['Duration'] = step_durations
        steps_df['Estimated_Duration'] = estimated_step_durations
        steps_df['Onset_Sample'] = step_onsets
        steps_df['Onset_Time'] = step_onsets / fr
        steps_df['Offset_Sample'] = step_offsets
        steps_df['Offset_Time'] = step_onsets / fr + np.array(step_durations) / fr
        steps_df['Detected_Offset_Time'] = step_offsets / fr
        steps_df['Stimulus'] = ['Step'] * len(step_durations)
        steps_df['SamplingRate'] = [fr] * len(step_durations)
    else:
        steps_df['Estimated_Duration'] = estimated_step_durations
        steps_df['Onset_Sample'] = step_onsets
        steps_df['Onset_Time'] = step_onsets / fr
        steps_df['Offset_Sample'] = step_offsets
        steps_df['Offset_Time'] = step_onsets / fr + np.array(estimated_step_durations) / fr
        steps_df['Detected_Offset_Time'] = step_offsets / fr
        steps_df['Stimulus'] = ['Step'] * len(estimated_step_durations)
        steps_df['SamplingRate'] = [fr] * len(estimated_step_durations)

    # COMBINED
    stimulus_protocol_unsorted = pd.concat([ramps_df, steps_df])
    stimulus_protocol = stimulus_protocol_unsorted.copy().sort_values(by=['Onset_Time']).reset_index(drop=True)

    # Test Detection
    if protocol_file:
        stim_count = 0
        total_presentations = 0
        for k in protocol_ramps['Duration']:
            d = stimulus_protocol['Duration'][stimulus_protocol['Stimulus'] == 'Ramp']
            c = np.sum(d == k)
            total_presentations += c
            stim_count += 1
            print(f'FOUND: {c} x {k} ms Ramps')

        for k in protocol_steps['Duration']:
            d = stimulus_protocol['Duration'][stimulus_protocol['Stimulus'] == 'Step']
            c = np.sum(d == k)
            total_presentations += c
            stim_count += 1
            print(f'FOUND: {c} x {k} ms Steps')
        print('----------')
        print(f'IN TOTAL: {stim_count} Stimulus Types ({len(protocol)} in protocol file)')
        print(f'IN TOTAL: {total_presentations} Events (Stimulus Presentations)')

    # STORE EVERYTHING TO DATAFRAME
    stimulus_final = pd.DataFrame()
    stimulus_final['Time'] = stimulus_time
    stimulus_final['Volt'] = stimulus
    uf.msg_box(f_header='INFO', f_msg='Stimulus detected !', sep='-')

    return stimulus_final, stimulus_protocol


def export_stimulus_file(s_file, s_protocol, export_protocol_name, export_stimulus_name):
    s_protocol.to_csv(export_protocol_name)
    # Match format to the stimulation file from Sutter-MView (MOM SETUP)
    s_file.to_csv(export_stimulus_name, sep='\t', decimal='.', header=None)


def detect_stimuli_from_trace(s_values, th_step=0.5, th_ramp=0.2, small_interval_th=0, smoothing_window=10,
                              show_helper_figure=False, compare=False):

    # Smooth Stimulus
    stimulus = np.convolve(s_values, np.ones(smoothing_window) / smoothing_window, mode='same')

    # Find Ramps and Steps in Stimulus Trace
    # Thresholding too small intervals
    # Min. distance between different stimuli in samples that is allowed!
    # This somehow corresponds to the stimulus intervals in the experiment (here: 30 secs)
    # This means it will ignore any stimulus detection x secs after the first detection! So the first must be correct!
    # I guess this could be improved... However, if 'th_step' and 'th_ramp' are set appropriately this small interval
    # thresholding should not be necessary anyways.
    ramp_stimulus, step_onsets, step_offsets, ramp_onsets, ramp_offsets = find_ramps_and_steps(
        s=stimulus.copy(), f_th_step=th_step, f_th_ramp=th_ramp, interval_th=small_interval_th
    )

    if show_helper_figure:
        fig, axs = plt.subplots(2, 1)
        axs[0].set_title('Stimulus Diff (red) (Estimate good step threshold value)')
        axs[0].plot(np.diff(stimulus, append=0), 'k')
        axs[0].plot([0, len(stimulus)], [th_step, th_step], 'r--')
        axs[1].plot(stimulus, 'k')
        axs[1].plot(step_onsets, np.zeros(len(step_onsets)) + th_step, 'xb', markersize=10)
        axs[1].plot(step_offsets, np.zeros(len(step_offsets)) + th_step, 'xb', markersize=10)
        axs[1].plot(ramp_onsets, np.zeros(len(ramp_onsets)) + th_ramp, 'xg', markersize=10)
        axs[1].plot(ramp_offsets, np.zeros(len(ramp_offsets)) + th_ramp, 'xg', markersize=10)
        axs[1].plot([0, len(stimulus)], [th_ramp, th_ramp], 'r--')
        axs[1].set_title('Stimulus Trace')
        plt.show()



    # Determine ramp and step durations
    fr = 1000
    stimulus_time = uf.convert_samples_to_time(stimulus, fr=fr)
    estimated_step_durations = step_offsets - step_onsets
    estimated_ramp_durations = ramp_offsets - ramp_onsets

    if compare:
        protocol_steps = pd.DataFrame(compare)
        protocol_steps.columns = ['Duration']
        step_durations = compare_stimulus_duration(
            f_protocol=protocol_steps, f_estimated_durations=estimated_step_durations
        )

    # Combine all info into one df
    # # RAMPS
    # ramps_df = pd.DataFrame()
    #
    # ramps_df['Estimated_Duration'] = estimated_ramp_durations / 2
    # ramps_df['Onset_Sample'] = ramp_onsets
    # ramps_df['Onset_Time'] = ramp_onsets / fr
    # ramps_df['Offset_Sample'] = ramp_offsets
    # ramps_df['Offset_Time'] = ramp_onsets / fr + np.array(estimated_ramp_durations) / fr
    # ramps_df['Detected_Offset_Time'] = ramp_offsets / fr
    # ramps_df['Stimulus'] = ['Ramp'] * len(estimated_ramp_durations)
    # ramps_df['SamplingRate'] = [fr] * len(estimated_ramp_durations)

    # STEPS
    steps_df = pd.DataFrame()
    steps_df['Duration'] = estimated_step_durations
    steps_df['Estimated_Duration'] = estimated_step_durations
    steps_df['Onset_Sample'] = step_onsets
    steps_df['Onset_Time'] = step_onsets / fr
    steps_df['Offset_Sample'] = step_offsets
    steps_df['Offset_Time'] = step_onsets / fr + np.array(estimated_step_durations) / fr
    steps_df['Detected_Offset_Time'] = step_offsets / fr
    steps_df['Stimulus'] = ['Step'] * len(estimated_step_durations)
    steps_df['SamplingRate'] = [fr] * len(estimated_step_durations)

    # COMBINED
    # stimulus_protocol_unsorted = pd.concat([ramps_df, steps_df])
    stimulus_protocol_unsorted = steps_df
    stimulus_protocol = stimulus_protocol_unsorted.copy().sort_values(by=['Onset_Time']).reset_index(drop=True)

    # STORE EVERYTHING TO DATAFRAME
    stimulus_final = pd.DataFrame()
    stimulus_final['Time'] = stimulus_time
    stimulus_final['Volt'] = stimulus
    uf.msg_box(f_header='INFO', f_msg='Stimulus detected !', sep='-')

    return stimulus_final, stimulus_protocol


def detect_sound_stimuli_from_trace(s_values, th_step=0.5, smoothing_window=10,
                              show_helper_figure=False, compare=False):

    # Smooth Stimulus
    stimulus = np.convolve(s_values, np.ones(smoothing_window) / smoothing_window, mode='same')

    # Find Ramps and Steps in Stimulus Trace
    # Thresholding too small intervals
    # Min. distance between different stimuli in samples that is allowed!
    # This somehow corresponds to the stimulus intervals in the experiment (here: 30 secs)
    # This means it will ignore any stimulus detection x secs after the first detection! So the first must be correct!
    # I guess this could be improved... However, if 'th_step' and 'th_ramp' are set appropriately this small interval
    # thresholding should not be necessary anyways.
    # f_single, f_trains, single_pulses_onsets, single_pulses_offsets, train_pulses_onsets, train_pulses_offsets
    detection_results = find_single_steps_and_trains(s_values, th_step, 15000)
    single_pulses = np.array(detection_results[0])
    train_pulses = np.array(detection_results[1])
    single_pulses_onsets = np.array(detection_results[2])
    single_pulses_offsets = np.array(detection_results[3])
    train_pulses_onsets = np.array(detection_results[4])
    train_pulses_offsets = np.array(detection_results[5])
    train_first_pulses_onsets = np.array(detection_results[6])
    train_first_pulses_offsets = np.array(detection_results[7])
    if show_helper_figure:
        fig, axs = plt.subplots(2, 1)
        axs[0].set_title('Stimulus Diff (red) (Estimate good step threshold value)')
        axs[0].plot(np.diff(stimulus, append=0), 'k')
        axs[0].plot([0, len(stimulus)], [th_step, th_step], 'r--')
        # axs[1].plot(stimulus, 'k')
        axs[1].plot(single_pulses, 'darkblue', alpha=0.5)
        axs[1].plot(train_pulses, 'darkgreen', alpha=0.5)
        axs[1].plot(single_pulses_onsets, np.zeros(len(single_pulses_onsets)) + th_step, 'xb', markersize=10)
        axs[1].plot(single_pulses_offsets, np.zeros(len(single_pulses_offsets)) + th_step, 'xb', markersize=10)
        axs[1].plot(train_pulses_onsets, np.zeros(len(train_pulses_onsets)) + th_step, 'xg', markersize=10)
        axs[1].plot(train_pulses_offsets, np.zeros(len(train_pulses_offsets)) + th_step, 'xg', markersize=10)
        axs[1].plot(train_first_pulses_onsets, np.zeros(len(train_first_pulses_onsets)) + th_step, 'xr', markersize=10)
        axs[1].plot(train_first_pulses_offsets, np.zeros(len(train_first_pulses_offsets)) + th_step, 'xr', markersize=10)

        axs[1].set_title('Stimulus Trace')
        plt.show()

    # Determine ramp and step durations
    fr = 1000
    stimulus_time = uf.convert_samples_to_time(stimulus, fr=fr)
    estimated_single_durations = single_pulses_offsets - single_pulses_onsets
    estimated_train_durations = train_pulses_offsets - train_pulses_onsets
    estimated_first_pulse_train_durations = train_first_pulses_offsets - train_first_pulses_onsets

    # SINGLE PULSES
    single_pulses_df = pd.DataFrame()
    single_pulses_df['Duration'] = estimated_single_durations
    single_pulses_df['Interval'] = np.zeros(len(estimated_single_durations))
    single_pulses_df['Onset_Sample'] = single_pulses_onsets
    single_pulses_df['Onset_Time'] = single_pulses_onsets / fr
    single_pulses_df['Offset_Sample'] = single_pulses_offsets
    single_pulses_df['Offset_Time'] = single_pulses_offsets / fr
    single_pulses_df['Stimulus'] = ['SinglePulse'] * len(estimated_single_durations)
    single_pulses_df['SamplingRate'] = [fr] * len(estimated_single_durations)
    
    # PULSE TRAINS
    idx = train_pulses_onsets >= train_first_pulses_onsets[-1]
    intervals = np.zeros(len(idx)) + 5
    intervals[idx] = 10

    train_pulses_df = pd.DataFrame()
    train_pulses_df['Duration'] = estimated_train_durations
    train_pulses_df['Interval'] = intervals
    train_pulses_df['Onset_Sample'] = train_pulses_onsets
    train_pulses_df['Onset_Time'] = train_pulses_onsets / fr
    train_pulses_df['Offset_Sample'] = train_pulses_offsets
    train_pulses_df['Offset_Time'] = train_pulses_offsets / fr
    train_pulses_df['Stimulus'] = ['TrainPulse'] * len(estimated_train_durations)
    train_pulses_df['SamplingRate'] = [fr] * len(estimated_train_durations)

    # FIRST PULSES IN TRAIN
    idx = train_first_pulses_onsets >= train_first_pulses_onsets[-1]
    intervals = np.zeros(len(idx)) + 5
    intervals[idx] = 10

    train_first_pulses_df = pd.DataFrame()
    train_first_pulses_df['Duration'] = estimated_first_pulse_train_durations
    train_first_pulses_df['Interval'] = intervals
    train_first_pulses_df['Onset_Sample'] = train_first_pulses_onsets
    train_first_pulses_df['Onset_Time'] = train_first_pulses_onsets / fr
    train_first_pulses_df['Offset_Sample'] = train_first_pulses_offsets
    train_first_pulses_df['Offset_Time'] = train_first_pulses_offsets / fr
    train_first_pulses_df['Stimulus'] = ['FirstTrainPulse'] * len(estimated_first_pulse_train_durations)
    # train_first_pulses_df['Stimulus'] = ['FirstTrainPulse_5s', 'FirstTrainPulse_5s', 'FirstTrainPulse_10s']
    train_first_pulses_df['SamplingRate'] = [fr] * len(estimated_first_pulse_train_durations)

    # COMBINED
    stimulus_protocol_unsorted = pd.concat([single_pulses_df, train_pulses_df, train_first_pulses_df])
    # stimulus_protocol_unsorted = single_pulses_df
    stimulus_protocol = stimulus_protocol_unsorted.copy().sort_values(by=['Onset_Time']).reset_index(drop=True)

    # STORE EVERYTHING TO DATAFRAME
    stimulus_final = pd.DataFrame()
    stimulus_final['Time'] = stimulus_time
    stimulus_final['Volt'] = stimulus
    uf.msg_box(f_header='INFO', f_msg='Stimulus detected !', sep='-')

    return stimulus_final, stimulus_protocol


# MAIN SCRIPT ----------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == '-estimate':
            estimate_round = True
            uf.msg_box('INFO', 'STARTING ESTIMATE MODE', '+')
        elif command == '-detect':
            estimate_round = False
            uf.msg_box('INFO', 'STARTING DETECTION MODE', '+')
        else:
            uf.msg_box('ERROR', 'INVALID ARGUMENT \n USE: -estimate or -detect', '+')
            exit()
    else:
        estimate_round = False
        uf.msg_box('ERROR', 'NO ARGUMENT WAS GIVEN \n USE: -estimate or -detect', '+')
        exit()

    select_single_file = True
    file_name = open_dir(select_single_file)
    file_dir = os.path.split(file_name)[0]
    rec_name = os.path.split(file_dir)[1]
    # Import stimulation file
    stimulation_file = pd.read_csv(f'{file_dir}/{rec_name}_stimulation.txt')
    # used_protocol_values = pd.read_csv('E:/CaImagingAnalysis/Paper_Data/Tapping/protocol_stimulus_values.csv')
    # used_protocol_values = pd.read_csv(f'{file_dir}/protocol_stimulus_values.csv')
    # Import raw data
    f_raw = uf.import_f_raw(f'{file_dir}/{rec_name}_raw.txt')

    # Estimate Frame Rate
    fr_rec = uf.estimate_sampling_rate(data=f_raw, f_stimulation=stimulation_file, print_msg=False)
    # Check if Recording has already been extended
    # file_list = [s for s in os.listdir(file_dir) if 'RECORDING_WAS_EXTENDED' in s]
    # if len(file_list) > 0:
    #     uf.msg_box('INFO', 'Recording is the extended version', '-')
    # else:
    #     uf.msg_box('WARNING', 'Recording seems to be not extended? Are You Sure to proceed?', '-')

    # Check if Stimulus Values must be inverted (if they are negative values)
    if stimulation_file['Volt'].max() <= 2:
        stimulation_file['Volt'] = stimulation_file['Volt'] * -1
        uf.msg_box('INFO', 'Stimulus Values have been inverted', sep='-')
    #
    # Detect Stimulus from voltage trace
    stimulation, protocol = detect_stimuli_from_trace(
        s_values=stimulation_file['Volt'].to_numpy(), th_step=0.1, th_ramp=0.2, small_interval_th=0,
        smoothing_window=10, show_helper_figure=estimate_round, compare=[2000])

    # Detect SOUND Stimulus from voltage trace
    # stimulation, protocol = detect_sound_stimuli_from_trace(
    #     s_values=stimulation_file['Volt'].to_numpy(), th_step=0.1,
    #     smoothing_window=10, show_helper_figure=estimate_round, compare=[2000])

    #
    # stimulation, protocol = detect_stimuli(
    #     s_values=stimulation_file['Volt'].to_numpy(), protocol_file=True,
    #     th_step=0.1, th_ramp=0.2, small_interval_th=0, smoothing_window=10, show_helper_figure=estimate_round,
    #     nils_wenke=False, protocol_values=used_protocol_values)

    # if estimate_round:
    #     print(protocol[protocol['Stimulus'] == 'SinglePulse'])
    #     print('')
    #     print(protocol[protocol['Stimulus'] == 'TrainPulse'])
    #     print('')
    #     print(protocol[protocol['Stimulus'] == 'FirstTrainPulse'])
    #     print('')
    #     print('INTERVALS:')
    #     print(protocol['Onset_Time'].diff().to_numpy())
    #     embed()
    #     exit()
    # else:
    #     stimulation.to_csv(f'{file_dir}/{rec_name}_stimulation_filtered.txt')
    #     protocol.to_csv(f'{file_dir}/{rec_name}_protocol.csv')
    #     uf.msg_box('INFO', 'STORED DETECTED PROTOCOL AND STIMULUS TO HDD', sep='+')
    #
    # exit()
    if estimate_round:
        print(protocol[protocol['Stimulus'] == 'Step'])
        print('')
        print(protocol[protocol['Stimulus'] == 'Ramp'])
        print('')
        print('INTERVALS:')
        print(protocol['Onset_Time'].diff().to_numpy())
        embed()
        exit()
    else:
        stimulation.to_csv(f'{file_dir}/{rec_name}_stimulation_filtered.txt')
        protocol.to_csv(f'{file_dir}/{rec_name}_protocol.csv')
        uf.msg_box('INFO', 'STORED DETECTED PROTOCOL AND STIMULUS TO HDD', sep='+')

    exit()
    #
    # import plottools.colors as c
    # # Compute Calcium Impulse Response Function (CIRF)
    # # taus_1 = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]
    # # taus = np.linspace(1, 20, 5)
    # taus_1 = [10, 15, 20, 25, 50, 200]
    # taus_2 = 2
    # amp = 1
    # regressors = []
    # binaries = []
    # for cirf_tau in taus_1:
    #     # cirf = uf.create_cif(fr_rec, tau=cirf_tau)
    #     cirf = uf.create_cif_double_tau(fr_rec, tau1=cirf_tau, tau2=taus_2, a=amp)
    #
    #     # Compute Regressor for the entire stimulus trace
    #     binary, reg, _, _ = uf.create_binary_trace(
    #         sig=f_raw, cirf=cirf, start=protocol['Onset_Time'], end=protocol['Offset_Time'],
    #         fr=fr_rec, ca_delay=0, pad_before=5, pad_after=20, low=0, high=1
    #     )
    #
    #     # Convolve Binary with cirf
    #     # spikes = np.diff(binary)
    #     # spikes[spikes<0] = 0
    #     # reg = np.convolve(spikes, cirf, 'full')
    #     binaries.append(binary)
    #     regressors.append(reg)
    #
    # ca_trace = f_raw['roi_7']
    # fbs = np.percentile(ca_trace, 5)
    # df_f = (ca_trace - fbs) / fbs
    # # Filter
    # ca_trace_filtered = uf.moving_average_filter(ca_trace, window=2)
    # df_filtered = uf.moving_average_filter(df_f, window=10)
    #
    # fbs2 = np.percentile(ca_trace_filtered, 5)
    # df_f2 = (ca_trace_filtered - fbs2) / fbs2
    # # df_f2 = uf.moving_average_filter(df_f2, window=2)
    #
    # # z_score = (ca_trace-np.mean(ca_trace))/np.std(ca_trace)
    # # z_score = uf.moving_average_filter(z_score, window=10) - np.min(z_score)
    # colors = c.palettes['muted']
    # color_names = list(colors.keys())
    # lightblue = c.lighter(colors['blue'], 0.4)
    # levels = np.linspace(0.25, 0.75, len(taus_1))
    #
    # plt.figure(figsize=(14, 4))
    # for i, r in enumerate(regressors):
    #     # plt.plot(r/np.max(r), color=c.lighter(colors['blue'], level), alpha=1)
    #     # plt.plot(r/np.max(r), color=colors[color_names[i]], alpha=1, label=f'tau: {taus[i]}')
    #     plt.plot(r/np.max(r) * 0.5, c.lighter(colors['red'], levels[i]), alpha=0.75, label=f'tau: {taus_1[i]}')
    #
    # # plt.plot(regressors[4]/np.max(regressors[1]), colors['black'], lw=2, alpha=1, label=f'tau: {taus[4]}')
    # # plt.plot(regressors[50]/np.max(regressors[50]), colors['black'], lw=2, alpha=1, label=f'tau: {taus[50]}')
    # plt.legend()
    #
    # # plt.plot(ca_trace/np.max(ca_trace), 'b', alpha=0.25)
    # # plt.plot(df_filtered/np.max(df_filtered), 'g', alpha=0.75)
    # plt.plot(binaries[0], 'k', alpha=0.2)
    # plt.plot(df_f2/np.max(df_f2), 'b', alpha=1)
    #
    # plt.show()
    #
    # exit()
    #
    # fr = 1000
    # t = np.arange(0, 10, 1/fr)
    #
    # tau1 = 0.7
    # tau2 = 1
    #
    # # for tau1 in [0.5, 1, 2, 5, 10, 20, 50]:
    # c1 = 1 - np.exp(-(t / tau1))
    # c2 = np.exp(-(t / tau2))
    # cif = c1 * c2
    # plt.plot(t, c1, 'b', alpha=0.25)
    # plt.plot(t, c2, 'r', alpha=0.25)
    # plt.plot(t, cif, 'k')
    #
    # plt.xlabel('Time [s]')
    # plt.show()
