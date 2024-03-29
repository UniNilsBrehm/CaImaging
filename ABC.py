import numpy as np
from IPython import embed
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from IPython import embed
import more_itertools
from scipy.interpolate import interp1d
import time as clock
import pymc as pm
# import emcee


def lif_ode():
    return

def df_over_f_static_percentile(sig, p=5):
    # Method 1
    base_f = np.percentile(sig, p, axis=0)
    df = (sig - base_f) / base_f
    # Method 2
    # sorted_values = np.sort(sig)
    # idx = int(len(sig) * (p / 100))+1
    # # base_f2 = np.median(sorted_values[0:idx])
    # base_f2 = sorted_values[idx]
    # df2 = (sig - base_f2) / base_f2

    return df, base_f


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
        th_limit = int(win_size * (per/100))
        output = []
        for w in win:
            sorted_win = np.sort(w)
            output.append(np.median(sorted_win[0:th_limit]))

    return output


def compute_df_over_f(f_values, window_size, per=5, fast=True):
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
    if window_size == 0:
        # Use static percentile method
        f_df, fbs = df_over_f_static_percentile(sig=f_values, p=per)
        f_rois = f_values.keys()
    else:
        if (window_size % 2) > 0:  # is it odd?
            # Then make it an even number:
            window_size = window_size + 1
        # Compute base line fb
        f_rois = f_values.keys()
        fbs = pd.DataFrame().reindex_like(f_values)
        for roi_nr in f_rois:
            data = f_values[roi_nr]
            fbs[roi_nr] = sliding_percentile(sig=data, win_size=window_size, per=per, fast_method=fast)

        # Compute delta f over f
        f_df = (f_values - fbs) / fbs
    return f_df, f_rois, fbs


def convolve_spikes(f_time, spike_trace, f_cirf):
    f_spikes_binary = np.zeros(len(f_time))
    for f_spk in spike_trace:
        # Compute Spikes Binary
        f_spikes_binary[(f_time == f_spk)] = 1
        # Compute Calcium Fluorescence Trace
    f_cat = np.convolve(f_spikes_binary, f_cirf, 'full')[:len(f_spikes_binary)]
    return f_cat


def create_cif_double_tau(tau1, tau2, a, dt):
    t_max = tau2 * 5  # in sec
    t_cif = np.arange(0, t_max, dt)
    cif = a * (1 - np.exp(-(t_cif/tau1)))*np.exp(-(t_cif/tau2))
    return cif


def get_stimulus(stimulation_dir, stimulus_strength):
    protocol = pd.read_csv(stimulation_dir, index_col=0)

    dt = 0.001
    time_max = np.round(protocol['Offset_Time'].max() + 10)
    time = np.arange(-0.2, time_max + dt, dt)
    stimulus = np.zeros(len(time)) + 0
    for t_on, t_off in zip(protocol['Onset_Time'], protocol['Offset_Time']):
        stimulus[(time >= t_on) & (time < t_off)] = stimulus_strength

    return time, stimulus


def convert_samples_to_time(sig, fr):
    t_out = np.linspace(0, len(sig) / fr, len(sig))
    return t_out


def linear_model(xx, yy, norm=True):
    if len(xx.shape) == 1:
        reg_xx = xx.reshape(-1, 1)
    elif len(xx.shape) == 2:
        reg_xx = xx
    else:
        print('ERROR: Wrong x input')
        return 0, 0, 0

    if norm:
        reg_xx = reg_xx / np.max(reg_xx)
        yy = yy / np.max(yy)

    # Linear Regression
    l_model = LinearRegression().fit(reg_xx, yy)
    # Slope (y = a * x + c)
    slope = l_model.coef_[0]
    # R**2 of model
    f_r_squared = l_model.score(reg_xx, yy)
    return f_r_squared, slope


def lif_for_fitting(alpha_slow):
    # The model will take approx. 1 sec to run
    # Get Stimulus
    stimulus_intensity = 2
    _, stimulus = get_stimulus(stimulation_dir=f'{base_dir}{recording_name}_protocol.csv',
                               stimulus_strength=stimulus_intensity)

    f_time = time
    alpha_fast = 0.1
    slow_dependency = 0.5
    taum = 0.01
    taua_slow = 15
    taua_fast = 0.1
    cirf_tau_1 = 0.1
    cirf_tau_2 = 8
    # slow_dependency = 0
    rng = np.random
    noisedv = 0.001
    vreset = 0.0
    vthresh = 1.0
    noiseda = 0.001
    # taum = 0.01
    tref = 0.003
    dt = f_time[1] - f_time[0]  # time step
    noisev = rng.randn(len(stimulus)) * noisedv / np.sqrt(dt)  # properly scaled voltage noise term
    noisea = rng.randn(len(stimulus)) * noiseda / np.sqrt(dt)  # properly scaled adaptation noise term

    # Compute Calcium Impulse Response Function
    # cirf_tau_1 = 0.04
    # cirf_tau_2 = 4
    cirf_time = np.arange(0, cirf_tau_2 * 5, dt)
    f_cirf = create_cif_double_tau(tau1=cirf_tau_1, tau2=cirf_tau_2, dt=dt, a=1)

    # initialization:
    # np.seterr('raise')
    tn = f_time[0]
    V = rng.rand() * (vthresh - vreset) + vreset
    A_fast = 0.0
    A_slow = 0.0

    # integration:
    spikes = []
    # membrane_voltage = []
    # adaptation_slow = []
    # adaptation_fast = []

    for k in range(len(stimulus)):
        # membrane_voltage.append(V)  # store membrane voltage value
        # adaptation_slow.append(A_slow)
        # adaptation_fast.append(A_slow)

        if f_time[k] < tn:
            continue  # no integration during refractory period
        V += (-V - A_slow - A_fast + stimulus[k] + noisev[k]) * dt / taum  # membrane equation
        A_fast += (-A_fast + noisea[k]) * dt / taua_fast  # fast adaptation dynamics
        A_slow += (-A_slow + noisea[k]) * dt / taua_slow  # slow adaptation dynamics

        if V > vthresh:  # threshold condition
            V = vreset  # voltage reset
            A_fast += alpha_fast / taua_fast  # fast adaptation increment
            if (A_slow > 0.001) & (slow_dependency > 0):
                A_slow += A_slow * slow_dependency  # slow adaptation increment
            else:
                A_slow += alpha_slow / taua_slow  # slow adaptation increment
            tn = f_time[k] + tref  # refractory period
            spikes.append(f_time[k])  # store spike time

    # Convolve Spikes with Calcium Impulse Response Function
    conv = convolve_spikes(f_time=f_time, spike_trace=spikes, f_cirf=f_cirf)
    # Compute z score
    conv_z = (conv - np.mean(conv)) / np.std(conv)
    return conv_z


def model_function(a, tau1, tau2):
    # y = (1-np.exp(x*a/tau)) + np.exp(x*a/tau)
    y = a * (1 - np.exp(-(x/tau1)))*np.exp(-(x/tau2))
    return y


def compute_observed_data(a, tau1, tau2):
    a = a + (np.random.randn() * 0.1)
    tau1 = tau1 + (np.random.randn() * 0.1)
    tau2 = tau2 + (np.random.randn() * 0.1)
    y = model_function(a, tau1, tau2)
    noise = np.random.randn(len(y)) * 0.01
    y = y + noise
    return y, a, tau1, tau2


def error_function(param_list):
    # unpack the parameter list
    alpha_slow = param_list
    # run the model with the new parameters, returning the info we're interested in
    # result = model_function(a, tau1, tau2)
    result = lif_for_fitting(alpha_slow)

    # return the sum of the squared errors
    return sum((result - ca_trace_recording) ** 2)


if __name__ == '__main__':
    t0 = clock.perf_counter()
    recording_name = '220525_05_01'
    base_dir = f'E:/CaImagingAnalysis/Paper_Data/Sound/Duration/220520_DOB/{recording_name}/'
    # base_dir = f'E:/CaImagingAnalysis/Paper_Data/Habituation/{recording_name}/'
    save_dir = f'E:/CaImagingAnalysis/Paper_Data/lif_model/fitting/'

    # Load Data
    time, stimulation = get_stimulus(stimulation_dir=f'{base_dir}{recording_name}_protocol.csv', stimulus_strength=1)
    f_raw_dir = f'{base_dir}{recording_name}_raw.txt'
    f_raw = pd.read_csv(f_raw_dir)

    # Compute delta f over f for recording trace
    df_f, f_rois, fbs = compute_df_over_f(f_raw, window_size=200, per=5, fast=True)
    roi = 'roi_25'
    ca_trace_recording = df_f[roi]
    fr_rec = 2.0345147125756804

    # Interpolation Ca Recording Trace for Linear Model
    time_recording = convert_samples_to_time(ca_trace_recording.to_numpy(), fr=fr_rec)
    f = interp1d(time_recording, ca_trace_recording, kind='linear', bounds_error=False, fill_value=0)
    ca_trace_recording = f(time)
    # Compute z score
    ca_trace_recording = (ca_trace_recording - np.mean(ca_trace_recording)) / np.std(ca_trace_recording)

    embed()
    exit()
    print('STARTING MCMC')
    # Minimize Error
    with pm.Model() as model:
        alpha = pm.Uniform('alpha', lower=0.1, upper=2)
        ca = pm.Poisson('ca', mu=alpha, observed=ca_trace_recording)
        trace = pm.sample(10**4)
    print('')
    print(f'This took: {(clock.perf_counter() - t0)/60} mins')
    print('')
    embed()
    exit()
    # # Linear Regression Model
    # r2, slope = linear_model(model_data, ca_trace_recording, norm=False)
    # rmse = np.sqrt(sum((model_data - ca_trace_recording) ** 2) / len(ca_trace_recording))
    #
    # print('')
    # print(f'R²: {r2}, slope: {slope}')
    # print(f'RMSE: {rmse}')
    #
    # plt.plot(time, ca_trace_recording, 'k')
    # plt.plot(time, model_data, 'r')
    # plt.show()
    # embed()
    # exit()