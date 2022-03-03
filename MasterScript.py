import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import more_itertools
import seaborn as sns
from skimage import io
from scipy import signal
from IPython import embed
from mpl_point_clicker import clicker
from math import factorial
import random


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688  [Titel anhand dieser ISBN in Citavi-Projekt übernehmen]
    """
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def estimate_sampling_rate(data, f_stimulation):
    r""" Estimate the sampling rate via the total duration and sample count
        ----------
        data : pandas data frame, shape (N,)
            the values of all ROIs.
        f_stimulation : pandas data frame, shape (N,)
            stimulation recording (voltage trace).
        Returns
        -------
        fr : float
            the estimated sampling rate.
        Notes
        -----
        the stimulation data frame needs a column called 'Time' with sample time points.
    """
    max_time = f_stimulation['Time'].max()
    fr = len(data) / max_time
    return fr


def import_f_values(data_path):
    f_values = pd.read_csv(data_path)
    # Drop unneeded column:
    f_values = f_values.drop([f_values.keys()[0]], axis=1)
    return f_values


def sliding_percentile(sig, win_size, per, fast_method):
    r""" Compute a signal base line based on the (5 %) percentile of a  sliding window.
            ----------
            sig : pandas data frame, shape (N,)
                the values of all ROIs.
            win_size : float or integer
                sliding window size in samples.
            per : float
                percentile value (e.g. 5 %)
            fast_method: Boolean
                if True, use the fast method based on numpy percentile function
                if False, use slow method based on the median value of the smallest 5 % values in the window
            Returns
            -------
            output : array_like
                the estimated base line of the input signal.
            Notes
            -----
        """
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


def compute_df_over_f(f_values, window_size):
    r""" Compute delta F over F values of the raw fluorescence input signal.
            ----------
            f_values : pandas data frame
                the raw fluorescence values of all ROIs.
            window_size : float or integer
                window size used for base line estimate (in seconds).
            Returns
            -------
            f_df : pandas data frame
                the estimated delta F over F values of all ROIs.
            f_rois : array_like
                Roi names (e.g. 'Mean1') from imagej
            fbs: list of all fbs estimates  for all ROIs.

            Notes
            -----
            f_values: Each Column is one ROI with its mean GCaMP fluorescence values
            Since window size must be an even number, the function will turn an odd number into the next even number.
        """
    # Check if window size in samples is an even number
    if (window_size % 2) > 0:  # is it odd?
        # Then make it an even number:
        window_size = window_size + 1
    # Compute base line fb
    f_rois = f_values.keys()
    fbs = pd.DataFrame().reindex_like(f_values)
    for roi_nr in f_rois:
        data = f_values[roi_nr]
        fbs[roi_nr] = sliding_percentile(sig=data, win_size=window_size, per=0.05, fast_method=True)

    # Compute delta f over f
    f_df = (f_values - fbs) / fbs
    return f_df, f_rois, fbs


def low_pass_filter(sig, order, cutoff, fs, d_fs=1000):
    cutoff_filter = cutoff / (d_fs/fs)
    sos = signal.butter(order, cutoff_filter, 'lp', fs=fs, output='sos')
    lpf = signal.sosfilt(sos, sig)
    return lpf


def export_delta_f_over_f(data, export_path, win_time, fr, f_filter='none'):
    # Compute window samples
    win_samples = int(win_time * fr)
    # Compute smoothing window size
    in_seconds = 2
    smoothing_window_size = int(in_seconds * fr)
    if (smoothing_window_size % 2) == 0:
        smoothing_window_size = smoothing_window_size + 1
    # Compute delta f over f:
    delta_f, rois, fbs = compute_df_over_f(data, win_samples)

    if f_filter == 'lp':
        # Low Pass Filter delta f values:
        df_filtered = pd.DataFrame(low_pass_filter(delta_f, order=4, cutoff=200, fs=sampling_rate, d_fs=1000), columns=rois)
    elif f_filter == 'sg':
        # Use Savitzky–Golay filter
        f_roi_names = delta_f.keys().to_numpy()
        df_filtered = delta_f.copy()
        for kk in range(len(f_roi_names)):
            df_filtered[f_roi_names[kk]] = savitzky_golay(y=delta_f[f_roi_names[kk]].to_numpy(),
                                                          window_size=smoothing_window_size, order=1)
    else:
        df_filtered = delta_f
    # Store to HDD
    df_filtered.to_excel(f'{export_path}.xlsx')
    return df_filtered


def samples_to_time(sig, fr):
    t_out = np.linspace(0, len(sig) / fr, len(sig))
    return t_out


def plot_ap_vs_pa(ap, pa, cells):
    # figsize = (width, height)
    fig, axs = plt.subplots(len(cells), 1, figsize=(4, 10), facecolor='w', edgecolor='k')
    max_y = np.max([np.max(ap), np.max(pa)])
    for k in range(len(cells)):
        axs[k].plot(ap[f'Mean{cells[k]}'], 'r')
        axs[k].plot(pa[f'Mean{cells[k]}'], 'b')
        axs[k].set_ylim([0, max_y])
    axs[0].legend(['ap', 'pa'])
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_total_stimulation(data, stim, sr, f_roi):
    t_data = samples_to_time(data, sr)
    f_fig, f_axs = plt.subplots(2, 1, figsize=(8, 5), facecolor='w', edgecolor='k')
    f_axs[0].plot(stim['Time'], stim['Volt'], 'k')
    f_axs[1].plot(t_data, data[f'Mean{f_roi}'], 'b')
    plt.show()


def text_window(str_text, inner_frame):
    inner_frame_symbols = '+' * inner_frame
    print('')
    print('+' * (len(str_text) + (inner_frame + 1) * 2))
    print(f'{inner_frame_symbols} {str_text} {inner_frame_symbols}')
    print('+' * (len(str_text) + (inner_frame + 1) * 2))
    print('')


def combine_stimulation_sweeps(f_stimulation, f_protocols):
    # Combine the stimulation files and protocol files of all sweeps into one data frame
    # INPUTS:
    # f_stimulation: list of all stimulation data frames (txt files)
    # f_protocols: list of all protocol data frames (cvs files)

    # Take the first sweep and put it into 'total_stimulation'
    total_stimulation = [f_stimulation[0]]
    stimulus_time_resolution = 0.001

    # Combine all single files to one data array
    for kk, v in enumerate(f_stimulation):
        if kk > 0:
            last_entry = total_stimulation[kk - 1]['Time'].iloc[-1] + stimulus_time_resolution
            new_time = f_stimulation[kk]['Time'] + last_entry
            # Replace original time with on going time:
            f_dummy = v.copy()
            f_dummy['Time'] = new_time
            total_stimulation.append(f_dummy)

        # This are now all stimuli of all sweeps combined into one continuous data frame
        # (as if it would be one ongoing recording)
    f_stimulation = pd.concat(total_stimulation)
    # Reset index
    f_stimulation = f_stimulation.reset_index()
    f_stimulation = f_stimulation.drop(['index'], axis=1)

    # Combine all stimulus protocol logs:
    sweep_names = []
    for ii in range(len(f_protocols)):
        sweep_names.append(f'sweep{ii + 1}')
    f_protocols = pd.concat(f_protocols, keys=sweep_names)
    return f_stimulation, f_protocols


def find_stimulus_time(volt_threshold, f_stimulation, mode='above'):
    # Find stimulus time points
    if mode == 'below':
        threshold_crossings = np.diff(f_stimulation['Volt'] < volt_threshold, prepend=False)
    else:
        threshold_crossings = np.diff(f_stimulation['Volt'] > volt_threshold, prepend=False)

    # Get Upward Crossings
    f_upward = np.argwhere(threshold_crossings)[::2, 0]  # Upward crossings
    # Remove to small intervals
    f_upward = thresholding_small_intervals(f_upward, f_stimulation)
    # Get Downward Crossings
    f_downward = np.argwhere(threshold_crossings)[1::2, 0]  # Downward crossings
    # Remove to small intervals
    f_downward = thresholding_small_intervals(f_downward, f_stimulation)
    return f_downward, f_upward


def thresholding_small_intervals(data, f_stimulation):
    # Threshold for too small intervals
    threshold_intervals = int(np.mean(np.diff(data)) / 4)
    idx = np.diff(data) > threshold_intervals
    idx = np.insert(idx, 0, True)
    stimulus_index_onset_points = data[idx]
    stimulus_onset_times = f_stimulation['Time'].iloc[stimulus_index_onset_points]
    return stimulus_onset_times, stimulus_index_onset_points


def read_stimulation_files(f_stimulation_files, f_stimulation_path, f_protocol_files, f_protocol_path,
                           f_static_path, f_static_files):
    f_stimulation = []
    f_protocols = []
    f_static_settings = []
    for kk in range(len(f_stimulation_files)):
        dummy_stimulation = pd.read_csv(f'{f_stimulation_path}{f_stimulation_files[kk]}', sep='\s+', decimal=',',
                                        header=None, names=['Time', 'Volt'])
        dummy_protocols = pd.read_excel(f'{f_protocol_path}{f_protocol_files[kk]}')
        if f_static_files:
            dummy_statics = pd.read_csv(f'{f_static_path}{f_static_files[kk]}', sep=';')
            f_static_settings.append(dummy_statics)
        else:
            f_static_settings.append([])

        # Drop unneeded column
        dummy_protocols = dummy_protocols.drop([dummy_protocols.keys()[0]], axis=1)
        f_protocols.append(dummy_protocols)
        f_stimulation.append(dummy_stimulation)
    return f_stimulation, f_protocols, f_static_settings


def check_static_settings(f_statics):
    # Check if all data frames in list "f_statics" are equal
    f_dummy = []
    for kk in range(len(f_statics)):
        f_dummy.append(f_statics[0].equals(f_statics[kk]))
    if all(f_dummy):
        return True
    else:
        return False


def plot_traces(data, f_stimulation, fr, im, limits, selected_rois):

    # Check ROIs count:
    roi_count = data.shape[1]
    roi_names = data.keys().to_numpy()
    if len(selected_rois) > 0:
        roi_names = roi_names[np.array(selected_rois) - 1]
        roi_count = len(roi_names)
    more_than_ten = False
    if roi_count > 10:
        # if there are more than 10 ROIs, randomly pick 10
        more_than_ten = True
        roi_names = random.sample(list(roi_names), 10)
        roi_count = 10

    t_rois = samples_to_time(data['Mean1'], fr)
    f_fig, f_axs = plt.subplots(roi_count+1, 1, figsize=(5, 8), sharex=True)  # fig size = (width, height)
    # Plot Stimulus Trace
    f_axs[0].plot(f_stimulation['Time'], f_stimulation['Volt'], 'r')
    f_axs[0].plot(f_stimulation['Time'], [2.5] * len(f_stimulation['Time']), 'k--', linewidth=0.5)

    f_axs[0].set_xticks([])
    f_axs[0].set_yticks([])
    f_axs[0].set_ylabel('Stim', rotation=0, labelpad=20)

    if len(limits) > 1:
        y_max_lim = limits[1]
        y_min_lim = limits[0]
    else:
        y_max_lim = np.max(np.max(data))
        y_min_lim = np.min(np.min(data))
    # Plot ROIs
    for kk in range(len(roi_names)):
        f_axs[kk+1].plot(t_rois, data[roi_names[kk]], 'k', linewidth=0.8)
        f_axs[kk+1].plot(t_rois, [0] * len(t_rois), 'k--', linewidth=0.5)
        f_axs[kk+1].set_xticks([])
        f_axs[kk+1].set_yticks([])
        # f_axs[kk+1].set_ylabel(f'ROI {kk+1}', rotation=0, labelpad=20)
        f_axs[kk+1].set_ylabel(f'ROI {roi_names[kk][4:]}', rotation=0, labelpad=20)

        # f_axs[kk+1].set_ylim([y_min_lim, y_max_lim])
    f_axs[-1].set_xticks(np.arange(0, np.round(np.max(t_rois)), 100))
    f_axs[-1].set_xlabel('Time [s]')
    sns.despine()

    # Plot ROIs
    if len(im) > 0:
        plt.figure(figsize=(6, 6))
        plt.imshow(im)
        sns.despine()
    plt.tight_layout()
    plt.show()


def compute_z_scores(data):
    return (data - data.mean()) / data.std()


def plot_stimulus_detection(f_stimulation, onset_times, f_th):
    d = onset_times[0][0]
    u = onset_times[1][0]
    f_range = [-2.5, 2.5]
    plt.plot(f_stimulation['Time'], f_stimulation['Volt'], 'k')
    plt.plot(f_stimulation['Time'], [f_th] * len(f_stimulation['Time']), 'g--')
    for kk in range(len(u)):
        plt.plot([u.iloc[kk], u.iloc[kk]], [f_range[0], f_range[1]], 'r--')
    for kk in range(len(d)):
        plt.plot([d.iloc[kk], d.iloc[kk]], [f_range[0], f_range[1]], 'b--')
    plt.show()


def manual_selection(data, sort_samples):
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(data)
    f_clicker = clicker(ax, ["event"], markers=["x"])
    plt.show()
    dummy = f_clicker.get_positions()
    f_samples = pd.DataFrame(dummy['event'])[0]

    if sort_samples:
        if (len(f_samples) % 2) == 0:
            f_stimulation_samples_sorted = []
            for k in np.arange(0, len(f_samples), 2):
                f_stimulation_samples_sorted.append([int(f_samples[k]), int(f_samples[k + 1])])
            f_samples = pd.DataFrame(f_stimulation_samples_sorted, columns=['Start', 'End'])
            print('FOLLOWING POSITIONS SELECTED:')
            print(f_samples)
        else:
            text_window('WARNING: NUMBER OF ENTRIES IS ODD, THEREFORE COULD NOT SORT INTO PAIRS', 2)
    else:
        print('FOLLOWING POSITIONS SELECTED:')
        print(f_samples)
    return f_samples


def convert_sample_idx(f_samples, fr_in, fr_out):
    f_samples_out = (f_samples / fr_in) * fr_out
    return f_samples_out.astype('int')


# NOTES:
# Directory Structure for Ca Imaging Data:
# Base Directory: Containing all Recordings
# Recording Date: Year-Mont-Day (exp. 20220215)
# Recording ID: Recording Number (exp. 02)
# -------------------------------------------------------
# logs: protocol log files
# rawdata: csv table containing raw gray values per ROI
# stimulation: text files containing analog stimulus recording
# tiffs: recording tiff files (registered)
# references: imagej rois and STD and MED Reference Images
#
# Exp. 'C:/Uni Freiburg/CaImagingAnalysis/WaterFlow/20220215/01/'

# ---- SCRIPT MAIN PART ------------------------------------------------------------------------------------------------

# SETTINGS
export_data = True
experiment_type = 'Tapping'
base_dir = 'C:/Uni Freiburg/CaImagingAnalysis/Tapping'
date_dir = '20220203'
recording_id = '01'
dir_path = f'{base_dir}/{date_dir}/{recording_id}'
protocol_path = f'{dir_path}/logs/protocols/'
static_path = f'{dir_path}/logs/statics/'
stimulation_path = f'{dir_path}/stimulation/'
rawdata_path = f'{dir_path}/rawdata/'
reference_path = f'{dir_path}/references/'

# sampling_rate = 8.138  # in Hz
df_base_line_window = 60  # in secs
stimulation_sampling_rate = 1000  # in Hz
# ----------------------------------------------------------------------------------------------------------------------

rawFiles = os.listdir(rawdata_path)
protocolFiles = os.listdir(protocol_path)
staticFiles = os.listdir(static_path)
stimulationFiles = os.listdir(stimulation_path)

# STIMULATION AND PROTOCOL LOG FILES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Read all sweeps (stimulation and protocols):
stimulation_list, protocol_list, static_settings = read_stimulation_files(
    f_stimulation_files=stimulationFiles,
    f_stimulation_path=stimulation_path,
    f_protocol_files=protocolFiles,
    f_protocol_path=protocol_path,
    f_static_files=staticFiles,
    f_static_path=static_path
)

# Combine them all in one data frame each:
stimulation, protocol = combine_stimulation_sweeps(stimulation_list, protocol_list)

# Check if static settings were the same for all sweeps:
if experiment_type == 'WaterFlow':
    if check_static_settings(static_settings):
        text_window('Static Settings are the same for all sweeps!', inner_frame=4)
        static_settings = static_settings[0]
    else:
        text_window('WARNING: Static Settings were not the same for all sweeps!', inner_frame=4)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import raw values, compute delta f over f and store to HDD, if this has not been done before.
# If it has then just import those values
# Import raw values:
try:
    raw_f_values = import_f_values(f'{rawdata_path}raw_values.csv')
except ValueError:
    raise ValueError('ERROR: DATA FILE COULD BE FOUND!')

# Estimate Sampling Rate
sampling_rate = np.round(estimate_sampling_rate(raw_f_values, stimulation), 2)

# If there is only the raw data:
if len(rawFiles) == 1:
    # Compute and Export delta f over f values:
    df = export_delta_f_over_f(data=raw_f_values, export_path=f'{rawdata_path}delta_f_over_f',
                               win_time=df_base_line_window, fr=sampling_rate, f_filter='sg')
    df_no_filter = export_delta_f_over_f(data=raw_f_values, export_path=f'{rawdata_path}delta_f_over_f_no_filter',
                                         win_time=df_base_line_window, fr=sampling_rate, f_filter='none')
    text_window('Converted Raw Fluorescence Values to Delta F Over F and stored to HDD', inner_frame=4)
else:
    # Import delta f over f values:
    text_window('IMPORTING DELTA F OVER F VALUES', inner_frame=4)
    df = pd.read_excel(f'{rawdata_path}delta_f_over_f.xlsx', index_col=0)
    df_no_filter = pd.read_excel(f'{rawdata_path}delta_f_over_f_no_filter.xlsx', index_col=0)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import ROI Reference Image:
# references_list = os.listdir(f'{reference_path}')
try:
    rois_image = plt.imread(f'{reference_path}rois.tif')
except FileNotFoundError:
    print('Could not find ROI reference image!')
    print('Image File name must be: rois.tif')
    rois_image = []
# Compute z scores
z_scores = compute_z_scores(df)
# Plot roi traces
# plot_traces(data=z_scores, f_stimulation=stimulation, fr=sampling_rate, im=rois_image, limits=[],
#            selected_rois=[1, 3, 4, 5])

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FIND STIMULATION TIME POINTS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
base_line = 2.5  # in Volt
# Change center to zero (2.5 to 0 V)
# stimulation_filtered = stimulation.copy()
# stimulation_centered['Volt'] = stimulation['Volt'] - 2.5
# Filter Stimulus
stimulation['Volt'] = savitzky_golay(y=stimulation['Volt'].to_numpy(), window_size=21, order=1)

manual_mark_stimulus = False
load_temp = True
file_name = f'{dir_path}/temp.pkl'
if manual_mark_stimulus:
    if load_temp:
        # Load temp from HDD
        open_file = open(file_name, "rb")
        sample_list = pickle.load(open_file)
        open_file.close()
        stimulation_idx = sample_list[0]
        stimulation_samples = sample_list[1]
        stimulation_times = sample_list[2]
        text_window('Imported marked stimulus times', 2)
    else:
        # Manually mark stimulus onsets and offsets
        stimulation_samples = manual_selection(data=stimulation['Volt'], sort_samples=True)
        stimulation_times = stimulation_samples / stimulation_sampling_rate
        # Convert stimulus sample idx into corresponding idx for delta f over f values (rois)
        stimulation_idx = convert_sample_idx(f_samples=stimulation_samples, fr_in=stimulation_sampling_rate, fr_out=sampling_rate)

        # Store (pickle) to HDD
        sample_list = [stimulation_idx, stimulation_samples, stimulation_times]
        open_file = open(file_name, "wb")
        pickle.dump(sample_list, open_file)
        open_file.close()
        text_window('Stored marked stimulus times to temp file on HDD', 2)

else:
    if load_temp:
        # Load temp from HDD
        open_file = open(file_name, "rb")
        sample_list = pickle.load(open_file)
        open_file.close()
        stimulation_idx = sample_list[0]
        stimulation_samples = sample_list[1]
        stimulation_times = sample_list[2]
        text_window('Imported marked stimulus times', 2)
    else:
        voltage_threshold = 0.2
        detection_downward, detection_upward = find_stimulus_time(volt_threshold=voltage_threshold,
                                                                  f_stimulation=stimulation, mode='above')
        # plot_stimulus_detection(f_stimulation=stimulation_filtered, onset_times=[detection_downward, detection_upward],
        #                         f_th=voltage_threshold)
        stimulation_samples = pd.DataFrame()
        stimulation_samples['Start'] = detection_upward[1]
        stimulation_samples['End'] = detection_downward[1]
        stimulation_times = pd.DataFrame()
        stimulation_times['Start'] = detection_upward[0] / stimulation_sampling_rate
        stimulation_times['End'] = detection_downward[0] / stimulation_sampling_rate
        stimulation_idx = convert_sample_idx(f_samples=stimulation_samples, fr_in=stimulation_sampling_rate, fr_out=sampling_rate)

        # Store (pickle) to HDD
        sample_list = [stimulation_idx, stimulation_samples, stimulation_times]
        open_file = open(file_name, "wb")
        pickle.dump(sample_list, open_file)
        open_file.close()
        text_window('Stored marked stimulus times to temp file on HDD', 2)

time_cutout = 30  # in secs
time_before = 10
samples_cutout = int(time_cutout * sampling_rate)
samples_before = int(time_before * sampling_rate)
tau_s = 3

# Add Time Line to delta f over f data frame
df['time'] = samples_to_time(df, sampling_rate)

if experiment_type == 'WaterFlow':
    # Create array with stimulus on coded as numbers (index in protocols)
    stimulus_id = np.zeros(len(df))
    stimulus_velocity = np.zeros(len(df))
    stimulus_duration = np.zeros(len(df))
    stimulus_direction = np.zeros(len(df))

    # Check if there are as many stimulus idx as in the protocol, or less:
    if len(stimulation_idx) <= len(protocol):
        for k in range(len(stimulation_idx)):
            # Get stimulus information and collect them
            stimulus_id[stimulation_idx['Start'][k]-samples_before:stimulation_idx['End'][k]+1] = k + 1
            stimulus_velocity[stimulation_idx['Start'][k]-samples_before:stimulation_idx['End'][k]+1] = protocol.iloc[k]['rate']
            stimulus_duration[stimulation_idx['Start'][k]-samples_before:stimulation_idx['End'][k]+1] = protocol.iloc[k]['duration']
            stimulus_direction[stimulation_idx['Start'][k]-samples_before:stimulation_idx['End'][k]+1] = protocol.iloc[k]['direction']

        # Add Columns to the data frame
        df['sID'] = stimulus_id
        df['velocity'] = stimulus_velocity
        df['duration'] = stimulus_duration
        df['direction'] = stimulus_direction

        # Transform sID and direction values to integers:S
        df = df.astype({"sID": int, "direction": int})

    # Create long (stacked) data frame
    df_stacked = df.melt(value_name='deltaf', var_name='roi', id_vars=['time', 'sID', 'velocity', 'duration', 'direction'])
    z_stacked = z_scores.melt(value_name='zscore', var_name='roi')
    df_no_filter_stacked = df_no_filter.melt(value_name='unfiltered', var_name='roi')
    df_stacked['zscore'] = z_stacked['zscore']
    df_stacked['unfiltered'] = df_no_filter_stacked['unfiltered']
else:
    # Create array with stimulus on coded as numbers (index in protocols)
    stimulus_id = np.zeros(len(df))
    stimulus_type = np.array([None] * len(df))
    stimulus_trial = np.zeros(len(df))
    stimulus_parameter = np.zeros(len(df))
    # Check if there are as many stimulus idx as in the protocol, or less:
    if len(stimulation_idx) <= len(protocol):
        for k in range(len(stimulation_idx)):
            # Compute exact cut out time
            cutout = int(samples_cutout - (stimulation_idx['End'][k]+1) + stimulation_idx['Start'][k])
            # Get stimulus information and collect them
            stimulus_id[stimulation_idx['Start'][k]-samples_before:stimulation_idx['End'][k]+1+cutout] = k + 1
            stimulus_trial[stimulation_idx['Start'][k]-samples_before:stimulation_idx['End'][k]+1+cutout] = int(protocol.index[k][0][5:])
            stimulus_type[stimulation_idx['Start'][k]-samples_before:stimulation_idx['End'][k]+1+cutout] = protocol.iloc[k]['stim']
            stimulus_parameter[stimulation_idx['Start'][k]-samples_before:stimulation_idx['End'][k]+1+cutout] = protocol.iloc[k]['parameter']

        # Add Columns to the data frame
        df['sID'] = stimulus_id
        df['type'] = stimulus_type
        df['parameter'] = stimulus_parameter
        df['trial'] = stimulus_trial

        # Transform sID and direction values to integers:S
        df = df.astype({"sID": int, "trial": int})

    # Compute trial times
    df['trial_time'] = 0
    for j in df['sID'].unique():
        if j > 0:
            df.loc[df['sID'] == j, 'trial_time'] = df['time'].loc[df['sID'] == j] - df['time'].loc[df['sID'] == j].iloc[0]
    # Create long (stacked) data frame
    df_stacked = df.melt(value_name='deltaf', var_name='roi',
                         id_vars=['time', 'trial_time', 'sID', 'type', 'parameter', 'trial'])
    z_stacked = z_scores.melt(value_name='zscore', var_name='roi')
    df_no_filter_stacked = df_no_filter.melt(value_name='unfiltered', var_name='roi')
    df_stacked['zscore'] = z_stacked['zscore']
    df_stacked['unfiltered'] = df_no_filter_stacked['unfiltered']

# Add new columns

# Store data to HDD:
df_stacked.to_csv(f'{dir_path}/data_long.csv')
# Store MetaData to HDD:
metadata_file_name = f'{dir_path}/MetaData.pkl'
metadata_list = [stimulation, protocol, sampling_rate, stimulation_sampling_rate]
metadata_dict = {'stimulation': stimulation, 'protocol': protocol, 'sampling_rate': sampling_rate,
                 'stimulus_sampling_rate': stimulation_sampling_rate}
open_file = open(metadata_file_name, "wb")
pickle.dump(metadata_dict, open_file)
open_file.close()

text_window('Data was organized in long format data frame and stored to HDD!', 4)
# select_data = df_stacked.loc[(df_stacked['sID'] == 2) & (df_stacked['roi'] == 'Mean2')]
# Plot some stuff
# p_data = df_stacked.loc[(df_stacked['sID'] == 1) & (df_stacked['roi'] == 'Mean1')]
