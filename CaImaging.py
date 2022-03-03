import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import more_itertools
from skimage import io
from scipy import signal
from IPython import embed


def conv_regresor(fr, tau, time_window, stim_dur, show_plot):
    # All inputs are in seconds (fr in Hz)
    # Define Stimulus regressor:
    fr = int(fr)
    regressor = np.zeros(fr * time_window)
    regressor_plot = np.zeros(fr * time_window)
    start = int(fr * time_window / 2)
    end = int(start + fr * stim_dur)
    regressor[start:end] = 1
    regressor_plot[0:int(stim_dur * fr)] = 1

    # numpy padding: add 5 zeros at the beginning ant none at the end:
    # np.pad(regressor, (5, 0), 'constant')

    # Define CIRF (Ca Impulse Response Function)
    tau_samples = fr * tau
    t_samples = np.arange(0, fr*time_window, 1)
    cirf = np.exp(-t_samples/tau_samples)
    # Convolution
    # convRegressor = np.convolve(regressor, cirf, 'full') / sum(cirf)
    # convRegressor_final = convRegressor[start-1: start - 1 + int(fr * time_window)]
    convRegressor_final = np.convolve(regressor, cirf, 'same') / sum(cirf)
    if show_plot:
        t_seconds = np.arange(0, time_window, 1/fr)
        # t_seconds_conv = np.arange(0, time_window*2, 1/fr)
        plt.figure()
        plt.plot(t_seconds, regressor_plot, 'k')
        plt.plot(t_seconds, cirf, 'b')
        plt.plot(t_seconds, convRegressor_final, 'r')
        plt.xlabel('Time [s]')
        plt.show()

    return convRegressor_final


def running_average(sig, n_samples):
    smooth = np.convolve(sig, np.ones(n_samples)/n_samples, 'same')
    return smooth


def samples_to_time(sig, fr):
    t_out = np.linspace(0, len(sig) / fr, len(sig))
    return t_out


def time_to_samples(sig):
    samples_out = np.linspace(0, len(sig), len(sig))
    return samples_out


def sliding_percentile(sig, win_size, per, fast_method):
    # Add as many value (edges) as win_size
    sig = np.pad(sig, (int(win_size/2), int((win_size/2)-1)), 'edge')
    # Create window (so that +- win_size centered around point of interest)
    win = list(more_itertools.windowed(sig, n=win_size, step=1))
    # Compute percentiles
    if fast_method:
        output = np.percentile(win, per, axis=1)
    else:
        th_limit = int(win_size * per)
        output = []
        for w in win:
            sorted_win = np.sort(w)
            output.append(np.median(sorted_win[0:th_limit]))

    return output


def df_over_f(sig, fbase):
    if len(sig) == len(fbase):
        df_out = (sig - fbase) / fbase
    else:
        raise ValueError('ERROR: Input Data Size Does Not Match!')
    return df_out


def low_pass_filter(sig, order, cutoff, fs, d_fs=1000):
    cutoff_filter = cutoff / (d_fs/fs)
    sos = signal.butter(order, cutoff_filter, 'lp', fs=fs, output='sos')
    lpf = signal.sosfilt(sos, sig)
    return lpf


def plot_response(f, roi_nr, stim, sr):
    # Plot complete Recording: Stimuli and F values
    rois = f.keys()[1:]
    # subtract 1 to roi (since idx start with 0: roi 1 == 0)
    data = f[rois[roi_nr-1]]
    t = samples_to_time(data, sr) / 60
    fig = plt.figure(figsize=(8, 2))
    plt.plot(t, data, 'k')
    plt.plot(stim['Time']/60, (stim['Volt'] / np.max(stim['Volt'])) * np.max(data), 'r', linewidth=0.5)
    plt.xlabel('Time')
    plt.ylabel('dF/F')
    plt.xlim([0, np.max(t)])
    plt.tight_layout()
    plt.show()
    plt.close(fig)


dir_path = 'C:/Uni Freiburg/CaImagingAnalysis/20220208/'
sweeps = False
load_tiff = False
waterflow = True
rawFiles = os.listdir(dir_path)
other_files = []
if sweeps:
    stimulation = []
    protocol_log = []
else:
    stimulation = pd.DataFrame()
    protocol_log = pd.DataFrame()
f_values = pd.DataFrame()


# Load Files
for file_name in rawFiles:
    if file_name[-15:-4] == 'CaRecording':
        tif_file_name = file_name
        if load_tiff:
            im = io.imread(f'{dir_path}{file_name}')
            print('Imported TIFF CA2+ RECORDING')
        else:
            print('FOUND TIFF CA2+ RECORDING (But not imported)')
    elif file_name[-15:-4] == 'stimulation':
        if sweeps:
            stimulation.append(pd.read_csv(f'{dir_path}{file_name}', sep='\s+', decimal=',', header=None,
                                           names=['Time', 'Volt']))
        else:
            stimulation = pd.read_csv(f'{dir_path}{file_name}', sep='\s+', decimal=',', header=None, names=['Time', 'Volt'])
        print('FOUND STIMULATION FILE')
    elif file_name[-8:-5] == 'log':
        if sweeps:
            protocol_log.append(pd.read_excel(f'{dir_path}{file_name}'))
        else:
            protocol_log = pd.read_excel(f'{dir_path}{file_name}')
        print('FOUND PROTOCOL LOG FILE')
    elif file_name == 'Fvalues.csv':
        f_values = pd.read_csv(f'{dir_path}{file_name}')
    else:
        other_files.append(file_name)
    if waterflow:
        if file_name == 'Fvalues_ap.csv':
            f_values_ap = pd.read_csv(f'{dir_path}{file_name}')
            print('FOUND AP FVALUES')
        if file_name == 'Fvalues_pa.csv':
            f_values_pa = pd.read_csv(f'{dir_path}{file_name}')
            print('FOUND PA FVALUES')

# Catch if files are not there:
if sweeps:
    if not protocol_log:
        raise ValueError('ERROR: NO PROTOCOL FILE COULD BE FOUND!')
    if not stimulation:
        raise ValueError('ERROR: NO STIMULATION FILE COULD BE FOUND!')
else:
    if protocol_log.empty:
        raise ValueError('ERROR: NO PROTOCOL FILE COULD BE FOUND!')
    if stimulation.empty:
        raise ValueError('ERROR: NO STIMULATION FILE COULD BE FOUND!')
if f_values.empty:
    raise ValueError('ERROR: NO Fluorescent Data FILE COULD BE FOUND!')


print('-------------------------------')
print('Also found this: ')
for others in other_files:
    print(others)
print('-------------------------------')


# If there are many sweeps
total_stimulation = [stimulation[0]]
stimulus_time_resolution = 0.001
if sweeps:
    # Combine all single files to one data array
    for k, v in enumerate(stimulation):
        if k > 0:
            last_entry = total_stimulation[k - 1]['Time'].iloc[-1] + stimulus_time_resolution
            new_time = stimulation[k]['Time'] + last_entry
            # Replace original time with on going time:
            dummy = v.copy()
            dummy['Time'] = new_time
            total_stimulation.append(dummy)

    # This are now all stimuli of all sweeps combined into one continuous data frame
    # (as if it would be one ongoing recording)
    stimulation = pd.concat(total_stimulation)
    # Combine all stimulus protocol logs:
    sweep_names = []
    for i in range(len(protocol_log)):
        sweep_names.append(f'sweep{i + 1}')
    protocol_log = pd.concat(protocol_log, keys=sweep_names)

# Estimate sampling rate
sampling_rate = np.round(len(f_values) / np.max(stimulation['Time']), 3)

print(f'Estimated 2P Sampling Rate: {sampling_rate} Hz')
print('-------------------------------')
# sampling_rate = 2

# Drop unneeded column
f_values = f_values.drop([f_values.keys()[0]], axis=1)

# Convert Raw Traces to Delta F over F
# Compute base line fb
rois = f_values.keys()
fbs = pd.DataFrame().reindex_like(f_values)

for roi_nr in rois:
    data = f_values[roi_nr]
    fbs[roi_nr] = sliding_percentile(sig=data, win_size=240, per=0.05, fast_method=True)

# Compute delta f over f
df = (f_values - fbs) / fbs

# Low Pass Filter df:
df_lpf = pd.DataFrame(low_pass_filter(df, order=4, cutoff=200, fs=sampling_rate, d_fs=1000), columns=rois)


time_axis_stimulation = stimulation['Time'].to_numpy()
# time_axis_f_values = np.arange(0, (len(f_values)-1) / sampling_rate, 1/sampling_rate)
time_axis_f_values = samples_to_time(df_lpf, fr=sampling_rate)

# Find stimulus time points
volt_threshold = 0.1
threshold_crossings = np.diff(stimulation['Volt'] > volt_threshold, prepend=False)
upward = np.argwhere(threshold_crossings)[::2, 0]  # Upward crossings
downward = np.argwhere(threshold_crossings)[1::2, 0]  # Downward crossings

# Threshold for too small intervals
threshold_intervals = int(np.mean(np.diff(upward)) / 4)
idx = np.diff(upward) > threshold_intervals
idx = np.insert(idx, 0, True)
stimulus_index_onset_points = upward[idx]
stimulus_onset_times = stimulation['Time'].iloc[stimulus_index_onset_points]

# Find this time point in dF values
# Convert time point into index of ROIs
idx_rois = np.round(sampling_rate * stimulus_onset_times)
idx_rois = idx_rois.astype('int')
stim_start = df_lpf['Mean1'][idx_rois]  # the keys are now the indices at which stimulus starts
idx_stim_start = stim_start.keys()

time_cutout = 40
samples_cutout = int(time_cutout * sampling_rate)
tau_s = 3

# --------------------------
# Plot complete Recording for one ROI
# plot_response(f=df_lpf, roi_nr=1, stim=stimulation, sr=sampling_rate)
#
# # Plot cutout stimuli:
# for b, v in enumerate(stimulus_index_onset_points):
#     a = stimulation['Volt'].iloc[v-500:v + 10000]
#     t_a = stimulation['Time'].iloc[v-500:v + 10000]
#     plt.plot(t_a, a)
# plt.show()

# --------------------------
# Get Responses of all cells to all stimuli
final_results = []
for k, i in enumerate(protocol_log.to_numpy()):
    # sn = f'{i[1]}{i[2]}-{i[0]}'
    sn = f'{i[1]}{i[2]}'
    sn_id = i[0]
    # results = {}
    rr = []
    for roi_nr in rois:
        # results[roi_nr] = df_lpf[roi_nr].iloc[idx_stim_start[k]:idx_stim_start[k] + samples_cutout]
        rr.append(df_lpf[roi_nr].iloc[idx_stim_start[k]:idx_stim_start[k] + samples_cutout].to_numpy())

    final_results.append(np.array(rr))
    # final_results.append(pd.DataFrame(np.transpose(rr), columns=rois))
# Dimensions: (stimuli presentation, rois, values)
final_results = np.array(final_results)

# Find the all trials of each stimuli
stim_names_unique = protocol_log[['stim', 'parameter']].drop_duplicates()
all_trials = []
for kk in range(len(stim_names_unique)):
    idx = stim_names_unique.iloc[kk] == protocol_log
    idx_trials = np.where(idx['parameter'] * idx['stim'])[0]
    all_trials.append(final_results[idx_trials])

# Dimensions: (stimuli, trial, rois, values)
all_trials = np.array(all_trials)
embed()
exit()

stacked = []
# Create a stacked pandas data frame
id_nr = 0
for stim_k, stim in enumerate(all_trials):
    for trial_k, trial in enumerate(stim):
        for roi_k, roi in enumerate(trial):
            for v_k, v in enumerate(roi):
                id_nr += 1
                # id : id value in roi : stimulus : trial : roi nr : value
                # stim_name_dummy = stim_names_unique.iloc[stim_k]['stim'] + str(stim_names_unique.iloc[stim_k]['parameter'])
                # row = [id_nr, v_k, stim_name_dummy, trial_k, roi_k, v]
                row = [id_nr, v_k, stim_k, trial_k, roi_k, v]
                # row = pd.DataFrame(row).transpose()
                stacked.append(row)
final_stack = pd.DataFrame(stacked, columns=['ID', 'ValID', 'Stimulus', 'Trial', 'ROI', 'Value'])

#
stimulus_names = []
for k, v in enumerate(stim_names_unique.to_numpy()):
    stimulus_names.append(v[0] + str(v[1]))

b = []
for k, v in enumerate(stim_names_unique.to_numpy()):
    a = final_stack.loc[(final_stack['Stimulus'] == k) & (final_stack['ROI'] == 1)]
    b.append(a['Value'].mean())
