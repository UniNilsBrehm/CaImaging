import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askdirectory
from tkinter.filedialog import askopenfilename
from tkinter import Tk
from IPython import embed
import analysis_util_functions as uf


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


def detect_stimuli(s_values):

    # Import Stimulus Protocols (hardcoded dir)
    # 6 stimulus types, each repeated for 5 times (total: 30)
    protocol_path = 'E:/CaImagingAnalysis/NilsWenke/'
    protocol_1 = pd.read_csv(f'{protocol_path}stimulus_protocol_1.csv')
    protocol_2 = pd.read_csv(f'{protocol_path}stimulus_protocol_2.csv')

    # Smooth Stimulus
    stimulus = uf.savitzky_golay(y=s_values, window_size=3, order=1)
    # Find Ramps and Steps in Stimulus Trace
    th_step = 0.5
    th_ramp = 0.2

    # Thresholding too small intervals
    # Min. distance between different stimuli in samples that is allowed!
    # This somehow corresponds to the stimulus intervals in the experiment (here: 30 secs)
    # This means it will ignore any stimulus detection x secs after the first detection! So the first must be correct!
    # I guess this could be improved... However, if 'th_step' and 'th_ramp' are set appropriately this small interval
    # thresholding should not be necessary anyways.
    small_interval_th = 0

    ramp_stimulus, step_onsets, step_offsets, ramp_onsets, ramp_offsets = find_ramps_and_steps(
        s=stimulus.copy(), f_th_step=th_step, f_th_ramp=th_ramp, interval_th=small_interval_th
    )

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

    # Determine ramp and step durations
    fr = 1000
    stimulus_time = uf.convert_samples_to_time(stimulus, fr=fr)
    estimated_step_durations = step_offsets - step_onsets
    estimated_ramp_durations = ramp_offsets - ramp_onsets

    # Compare estimated Step duration values to protocol files
    # STEPS
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
    ramps_df['Duration'] = ramp_matched_durations
    ramps_df['Estimated_Duration'] = estimated_ramp_durations / 2
    ramps_df['Onset_Sample'] = ramp_onsets
    ramps_df['Onset_Time'] = ramp_onsets / fr
    ramps_df['Offset_Sample'] = ramp_offsets
    ramps_df['Offset_Time'] = ramp_onsets / fr + np.array(ramp_matched_durations) / fr
    ramps_df['Detected_Offset_Time'] = ramp_offsets / fr
    ramps_df['Stimulus'] = ['Ramp'] * len(ramp_matched_durations)
    ramps_df['SamplingRate'] = [fr] * len(ramp_matched_durations)

    # STEPS
    steps_df = pd.DataFrame()
    steps_df['Duration'] = step_durations
    steps_df['Estimated_Duration'] = estimated_step_durations
    steps_df['Onset_Sample'] = step_onsets
    steps_df['Onset_Time'] = step_onsets / fr
    steps_df['Offset_Sample'] = step_offsets
    steps_df['Offset_Time'] = step_onsets / fr + np.array(step_durations) / fr
    steps_df['Detected_Offset_Time'] = step_offsets / fr
    steps_df['Stimulus'] = ['Step'] * len(step_durations)
    steps_df['SamplingRate'] = [fr] * len(step_durations)

    # COMBINED
    stimulus_protocol_unsorted = pd.concat([ramps_df, steps_df])
    stimulus_protocol = stimulus_protocol_unsorted.copy().sort_values(by=['Onset_Time']).reset_index(drop=True)

    # Test Detection
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


if __name__ == '__main__':
    select_single_file = False
    rec_dir = open_dir(select_single_file)
