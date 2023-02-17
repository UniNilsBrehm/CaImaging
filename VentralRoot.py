import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tkinter.filedialog import askdirectory
from tkinter.filedialog import askopenfile
from tkinter import Tk
from IPython import embed
from scipy.signal import hilbert
import scipy.signal as sig


def envelope(data, rate, freq=100.0):
    # Low pass filter the absolute values of the signal in both forward and reverse directions,
    # resulting in zero-phase filtering.
    sos = sig.butter(2, freq, 'lowpass', fs=rate, output='sos')
    filtered = sig.sosfiltfilt(sos, data)
    env = np.sqrt(2)*sig.sosfiltfilt(sos, np.abs(data))
    return filtered, env


def convert_to_secs(data, method=0):
    if method == 0:
        # INPUT: hhmmss.ms (163417.4532)
        s = str(data)
        in_secs = int(s[:2]) * 3600 + int(s[2:4]) * 60 + float(s[4:])
    elif method == 1:
        # INPUT: hh:mm:ss
        s = data
        in_secs = int(s[:2]) * 3600 + int(s[3:5]) * 60 + float(s[6:])
    else:
        in_secs = None
    return in_secs


def transform_ventral_root_recording():
    Tk().withdraw()
    base_dir = askdirectory()
    raw_data_dir = f'{base_dir}/rawdata'
    file_list = os.listdir(raw_data_dir)
    vr_files = [s for s in file_list if "ephys" in s]

    # Load files
    vr_data = []
    vr_time_secs = []
    for f_name in vr_files:
        print(f_name)
        dummy = pd.read_csv(f'{raw_data_dir}/{f_name}', sep='\t', header=None)
        vr_data.append(dummy)
        for i, v in enumerate(dummy.iloc[:, 3].to_numpy()):
            s = convert_to_secs(v)
            vr_time_secs.append(s)

    # Reset time so that it starts at 0
    vr_time_secs = np.array(vr_time_secs)
    vr_time_secs = vr_time_secs - vr_time_secs[0]

    # Concat all to one data frame
    vr_trace = pd.concat(vr_data).iloc[:, 0]
    # vr_time_stamps = pd.concat(vr_data).iloc[:, 1]

    # Load ca imaging recording
    # raw_file = [s for s in os.listdir(base_dir) if "raw_values" in s][0]
    # f_raw = pd.read_csv(f'{base_dir}/{raw_file}', index_col=0)
    # fr = 2
    # f_time_axis = np.linspace(0, len(f_raw) / fr, len(f_raw))
    # vr_time_axis = np.linspace(0, f_time_axis[-1], len(vr_trace))

    # Put all in one Data Frame
    vr_trace_export = pd.DataFrame(columns=['Time', 'Volt'])
    # Add the time in secs (not the timestamps)
    vr_trace_export['Time'] = vr_time_secs
    vr_trace_export['Volt'] = vr_trace.to_numpy()

    # Compute Envelope of VR Trace
    vr_fr = 10000
    vr_fil, vr_env = envelope(vr_trace_export['Volt'], vr_fr, freq=20.0)
    vr_env_export = pd.DataFrame(columns=['Time', 'Volt'])
    vr_env_export['Time'] = vr_trace_export['Time']
    vr_env_export['Volt'] = vr_env

    # Down-sample ventral root recording
    ds_factor = 64
    vr_trace_export_ds = vr_trace_export[::ds_factor]
    vr_env_export_ds = vr_env_export[::ds_factor]

    # Export to HDD
    vr_trace_export_ds.to_csv(f'{base_dir}/ventral_root_trace.csv', index=False)
    vr_env_export.to_csv(f'{base_dir}/ventral_root_envelope.csv', index=False)
    print('DONE')

    # Down-sampled plot
    # fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    # axs[0].set_title('Down-Sampled')
    # axs[0].plot(vr_trace_export_ds['Time'], vr_trace_export_ds['Volt'], 'r')
    # axs[0].plot(vr_env_export_ds['Time'], vr_env_export_ds['Volt'], 'k')
    #
    # axs[1].set_title('Original')
    # axs[1].plot(vr_trace_export['Time'], vr_trace_export['Volt'], 'b')
    # axs[1].plot(vr_trace_export['Time'], vr_fil, 'r')
    # axs[1].plot(vr_env_export['Time'], vr_env_export['Volt'], 'k')
    #
    # plt.show()


def ventral_root_stimulation():
    Tk().withdraw()
    base_dir = askdirectory()
    stimulus_data_dir = f'{base_dir}/stimulation'

    # Load Meta Data
    file_list = os.listdir(base_dir)
    meta_data_file_name = [s for s in file_list if 'meta_data.csv' in s][0]
    meta_data = pd.read_csv(f'{base_dir}/{meta_data_file_name}')
    time_zero = float(meta_data[meta_data['parameter'] == 'rec_vr_start']['value'].item())
    rec_duration = float(meta_data[meta_data['parameter'] == 'rec_vr_duration']['value'].item())

    file_list = os.listdir(stimulus_data_dir)
    # vr_files = [s for s in file_list if "ephys" in s]
    stimulus = dict()
    for f_name in file_list:
        stimulus[f_name[6:-4]] = pd.read_csv(f'{stimulus_data_dir}/{f_name}', sep='\t', header=None)
    stimulus_id = file_list[0][:5]
    idx = list(stimulus.keys())
    stimulus_df = pd.DataFrame()
    for stim_name in idx:
        # MOVING TARGET SMALL
        if stim_name == 'movingtargetsmall':
            # convert all timestamps to secs
            t_secs = []
            for i in range(stimulus[stim_name].shape[0]):
                t_secs.append(convert_to_secs(stimulus[stim_name][3].iloc[i][11:], method=1) - time_zero)
            t_secs = np.array(t_secs)

            # Get start and end times of the trials (via the large pause in between)
            d = np.diff(t_secs, append=0)
            idx = np.where(d > 15)
            trial_start = [t_secs[0]]
            trial_end = []
            for i in idx[0]:
                trial_end.append(t_secs[i])
                trial_start.append(t_secs[i+1])
            trial_end.append(t_secs[-1])

            # # get start and end time
            # s = stimulus[stim_name][3].iloc[0][11:]
            # e = stimulus[stim_name][3].iloc[-1][11:]
            # start_sec = convert_to_secs(s, method=1) - time_zero
            # end_sec = convert_to_secs(e, method=1) - time_zero
            # stimulus_df[stim_name] = [start_sec, end_sec]
            # stimulus_df = pd.concat([stimulus_df, pd.Series([start_sec, end_sec, stim_name]).to_frame().T], ignore_index=True)

            val = 1
            cc = 0
            for s, e in zip(trial_start, trial_end):
                stimulus_df = pd.concat([stimulus_df, pd.Series([s, e, val, f'{stim_name}_{cc}']).to_frame().T], ignore_index=True)
                cc += 1

        if stim_name == 'movingtargetlarge':
            # convert all timestamps to secs
            t_secs = []
            for i in range(stimulus[stim_name].shape[0]):
                t_secs.append(convert_to_secs(stimulus[stim_name][3].iloc[i][11:], method=1) - time_zero)
            t_secs = np.array(t_secs)

            # Get start and end times of the trials (via the large pause in between)
            d = np.diff(t_secs, append=0)
            idx = np.where(d > 15)
            trial_start = [t_secs[0]]
            trial_end = []
            for i in idx[0]:
                trial_end.append(t_secs[i])
                trial_start.append(t_secs[i+1])
            trial_end.append(t_secs[-1])

            val = 2
            cc = 0
            for s, e in zip(trial_start, trial_end):
                stimulus_df = pd.concat([stimulus_df, pd.Series([s, e, val, f'{stim_name}_{cc}']).to_frame().T], ignore_index=True)
                cc += 1

        # if stim_name == 'movingtargetlarge':
        #     # get start and end time
        #     s = stimulus[stim_name][3].iloc[0][11:]
        #     e = stimulus[stim_name][3].iloc[-1][11:]
        #     start_sec = convert_to_secs(s, method=1) - time_zero
        #     end_sec = convert_to_secs(e, method=1) - time_zero
        #     stimulus_df = pd.concat([stimulus_df, pd.Series([start_sec, end_sec, stim_name]).to_frame().T], ignore_index=True)

        if stim_name == 'flash':
            # convert all timestamps to secs
            t_secs = []
            for i in range(stimulus[stim_name].shape[0]):
                t_secs.append(convert_to_secs(stimulus[stim_name][1].iloc[i][11:], method=1) - time_zero)

            # t_secs = np.array(t_secs)
            t_secs = t_secs[3:]
            flash_dur = 4  # secs
            # The first entry is time when light turns on, then it turns off, then on etc.
            light_on_off = t_secs[::2]
            light_on_off.append(t_secs[-1])
            cc = 1
            for i in light_on_off:
                if (cc % 2) == 0:
                    state = 'OFF'
                    val = -1
                else:
                    state = 'ON'
                    val = 1
                stimulus_df = pd.concat([stimulus_df, pd.Series([i, i+flash_dur, val, f'{stim_name}_light_{state}']).to_frame().T],
                                        ignore_index=True)
                cc += 1

            # Somehow the first timestamp entry seems to be off... so take the second
            # s = stimulus[stim_name][1].iloc[1][11:]
            # e = stimulus[stim_name][1].iloc[-1][11:]
            # start_sec = convert_to_secs(s, method=1) - time_zero
            # end_sec = convert_to_secs(e, method=1) - time_zero
            # stimulus_df = pd.concat([stimulus_df, pd.Series([start_sec, end_sec, stim_name]).to_frame().T], ignore_index=True)

        if stim_name == 'grating':
            orientation = [0, 90, 180, 270]
            vals = [0.25, 0.5, 0.75, 1.0]
            # start_sec = []
            # end_sec = []
            cc = 0
            for o in orientation:
                idx_o = stimulus[stim_name].iloc[:, 1] == o
                s = stimulus[stim_name][idx_o][2].iloc[0][11:]
                # start_sec.append(convert_to_secs(s, method=1) - time_zero)
                start_sec = convert_to_secs(s, method=1) - time_zero
                e = stimulus[stim_name][idx_o][2].iloc[-1][11:]
                end_sec = convert_to_secs(e, method=1) - time_zero
                stimulus_df = pd.concat([stimulus_df, pd.Series([start_sec, end_sec, vals[cc], f'{stim_name}_{o}']).to_frame().T], ignore_index=True)
                cc += 1

        if stim_name == 'looming':
            val = 1
            s = stimulus[stim_name][3].iloc[0][11:]
            e = stimulus[stim_name][3].iloc[-1][11:]
            start_sec = convert_to_secs(s, method=1) - time_zero
            end_sec = convert_to_secs(e, method=1) - time_zero
            stimulus_df = pd.concat([stimulus_df, pd.Series([start_sec, end_sec, val, stim_name]).to_frame().T], ignore_index=True)

        if stim_name == 'looming_rev':
            val = -1
            s = stimulus[stim_name][3].iloc[0][11:]
            e = stimulus[stim_name][3].iloc[-1][11:]
            start_sec = convert_to_secs(s, method=1) - time_zero
            end_sec = convert_to_secs(e, method=1) - time_zero
            stimulus_df = pd.concat([stimulus_df, pd.Series([start_sec, end_sec, val, stim_name]).to_frame().T], ignore_index=True)

    stimulus_df.columns = ['start', 'end', 'value', 'stimulus']
    stimulus_df_sorted = stimulus_df.sort_values(by=['start'])
    print(stimulus_df_sorted)

    # Create Rectangular Signal
    dt = 0.001
    time_axis = np.arange(0, rec_duration, dt)
    rect_sig = np.zeros_like(time_axis)
    for k in range(stimulus_df_sorted.shape[0]):
        idx = (time_axis >= stimulus_df_sorted.iloc[k, :]['start']) & (time_axis < stimulus_df_sorted.iloc[k, :]['end'])
        rect_sig[idx] = stimulus_df_sorted.iloc[k, :]['value']
    rect_sig = rect_sig + 1
    # Store stimulus trace to HDD
    stimulus_trace = pd.DataFrame()
    stimulus_trace['Time'] = time_axis
    stimulus_trace['Volt'] = rect_sig

    stimulus_trace.to_csv(f'{base_dir}/stimulation_trace.csv', index=False)
    stimulus_df_sorted.to_csv(f'{base_dir}/stimulation_protocol.csv', index=False)
    print('Stimulus Protocol store to HDD')


def get_meta_data():
    olympus_setup_delay = 0.680  # Ahsan's Value
    time_stamp_key = '[Acquisition Parameters Common] ImageCaputreDate'
    time_stamp_key_ms = '[Acquisition Parameters Common] ImageCaputreDate+MilliSec'
    rec_duration_key = 'Time Per Series'
    rec_dt_key = 'Time Per Frame'
    Tk().withdraw()
    file_dir = askdirectory()
    img_rec_meta_data_file = [s for s in os.listdir(file_dir) if 'recording_file_metadata' in s][0]
    rec_meta_data = pd.read_csv(f'{file_dir}/{img_rec_meta_data_file}', sep='\t')
    rec_name = img_rec_meta_data_file[:11]

    time_stamp = rec_meta_data[rec_meta_data['Key'] == time_stamp_key]['Value'].values[0][12:-1]
    time_stamp_ms = rec_meta_data[rec_meta_data['Key'] == time_stamp_key_ms]['Value'].values[0]
    img_rec_start = int(time_stamp[:2]) * 3600 + int(time_stamp[3:5]) * 60 + int(time_stamp[6:]) + int(time_stamp_ms)/1000
    img_rec_duration = float(rec_meta_data[rec_meta_data['Key'] == rec_duration_key]['Value'].item()) / (1000*1000)
    img_rec_dt = float(rec_meta_data[rec_meta_data['Key'] == rec_dt_key]['Value'].item()) / (1000*1000)
    img_rec_fr = 1/img_rec_dt

    raw_data_dir = f'{file_dir}/rawdata'
    file_list = os.listdir(raw_data_dir)
    # Get first time stamp of first vr file
    vr_files = [s for s in file_list if "ephys" in s]
    vr_first = pd.read_csv(f'{raw_data_dir}/{vr_files[0]}', sep='\t', header=None)
    vr_last = pd.read_csv(f'{raw_data_dir}/{vr_files[-1]}', sep='\t', header=None)

    vr_start = convert_to_secs(vr_first[3].iloc[0])
    vr_end = convert_to_secs(vr_last[3].iloc[-1])
    print(f'VR RECORDING DURATION: {vr_end-vr_start:.3f} secs')

    # Meta Data File
    parameters_dict = {
        'rec_id': rec_name,
        'rec_img_start': img_rec_start,
        'rec_img_start_plus_delay': img_rec_start + olympus_setup_delay,
        'rec_img_duration': img_rec_duration,
        'rec_img_dt': img_rec_dt,
        'rec_img_fr': img_rec_fr,
        'rec_vr_start': vr_start,
        'rec_vr_end': vr_end,
        'rec_vr_duration': vr_end-vr_start
    }
    meta_data = pd.DataFrame(list(parameters_dict.items()), columns=['parameter', 'value'])
    meta_data.to_csv(f'{file_dir}/{rec_name}_meta_data.csv')
    print('Meta Data stored to HDD')


if __name__ == '__main__':
    print('')
    print('Type in the number of the function you want to use')
    print('')
    print('0: Get Meta Data')
    print('1: Get Stimuli')
    print('2: Transform Ventral Root Recording')
    print('')
    print('To exit type: >> exit')
    x = True
    while x:
        print('')
        usr = input("Enter: ")
        if usr == '0':
            get_meta_data()
        if usr == '1':
            ventral_root_stimulation()
        elif usr == '2':
            transform_ventral_root_recording()
        elif usr == 'exit':
            exit()
