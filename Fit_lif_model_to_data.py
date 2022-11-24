import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from IPython import embed
import pandas as pd
import more_itertools
from scipy.interpolate import interp1d
import time as clock
from scipy.stats.distributions import t as t_stats


def lif_with_outputs(time, alpha_slow, alpha_fast, slow_dependency):
    # Get Stimulus
    stimulus_intensity = 2
    _, stimulus = get_stimulus(stimulation_dir=f'{base_dir}{recording_name}_protocol.csv',
                               stimulus_strength=stimulus_intensity)

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
    dt = time[1] - time[0]  # time step
    noisev = rng.randn(len(stimulus)) * noisedv / np.sqrt(dt)  # properly scaled voltage noise term
    noisea = rng.randn(len(stimulus)) * noiseda / np.sqrt(dt)  # properly scaled adaptation noise term

    # Compute Calcium Impulse Response Function
    # cirf_tau_1 = 0.04
    # cirf_tau_2 = 4
    cirf_time = np.arange(0, cirf_tau_2 * 5, dt)
    f_cirf = create_cif_double_tau(tau1=cirf_tau_1, tau2=cirf_tau_2, dt=dt, a=1)

    # initialization:
    # np.seterr('raise')
    tn = time[0]
    V = rng.rand() * (vthresh - vreset) + vreset
    A_fast = 0.0
    A_slow = 0.0

    # integration:
    spikes = []
    membrane_voltage = []
    adaptation_slow = []
    adaptation_fast = []

    for k in range(len(stimulus)):
        membrane_voltage.append(V)  # store membrane voltage value
        adaptation_slow.append(A_slow)
        adaptation_fast.append(A_slow)
        if time[k] < tn:
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
            tn = time[k] + tref  # refractory period
            spikes.append(time[k])  # store spike time

    # Convolve Spikes with Calcium Impulse Response Function
    conv = convolve_spikes(f_time=time, spike_trace=spikes, f_cirf=f_cirf)
    return {'spikes': np.asarray(spikes), 'volt': np.asarray(membrane_voltage),
            'a_slow': np.asarray(adaptation_slow), 'a_fast': np.asarray(adaptation_fast), 'ca': conv,
            'time_cirf': cirf_time, 'cirf': f_cirf, 'cirf_tau_1': cirf_tau_1, 'cirf_tau_2': cirf_tau_2}


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


def fit_model(data_time, data):
    # TODO: - preset all parameters to a default value
    #       - then give lif function dict as parameters
    #       - check if parameter is in input dict, if yes replace default value with it
    #       - consider that different parameter ranges can lead to different results ...
    #       - how does curve_fit iterate through parameter bounds?
    #       - add initial guesses as well

    def lif_for_fitting(f_time, alpha_slow, alpha_fast, slow_dependency):
        # Get Stimulus
        stimulus_intensity = 2
        _, stimulus = get_stimulus(stimulation_dir=f'{base_dir}{recording_name}_protocol.csv',
                                   stimulus_strength=stimulus_intensity)

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
        dt = f_time[1] - f_time[0]                                # time step
        noisev = rng.randn(len(stimulus))*noisedv/np.sqrt(dt) # properly scaled voltage noise term
        noisea = rng.randn(len(stimulus))*noiseda/np.sqrt(dt) # properly scaled adaptation noise term

        # Compute Calcium Impulse Response Function
        # cirf_tau_1 = 0.04
        # cirf_tau_2 = 4
        cirf_time = np.arange(0, cirf_tau_2 * 5, dt)
        f_cirf = create_cif_double_tau(tau1=cirf_tau_1, tau2=cirf_tau_2, dt=dt, a=1)

        # initialization:
        # np.seterr('raise')
        tn = f_time[0]
        V = rng.rand()*(vthresh-vreset) + vreset
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
                continue                 # no integration during refractory period
            V += (-V - A_slow - A_fast + stimulus[k] + noisev[k])*dt/taum   # membrane equation
            A_fast += (-A_fast + noisea[k])*dt/taua_fast  # fast adaptation dynamics
            A_slow += (-A_slow + noisea[k]) * dt / taua_slow  # slow adaptation dynamics

            if V > vthresh:              # threshold condition
                V = vreset               # voltage reset
                A_fast += alpha_fast/taua_fast          # fast adaptation increment
                if (A_slow > 0.001) & (slow_dependency > 0):
                    A_slow += A_slow * slow_dependency  # slow adaptation increment
                else:
                    A_slow += alpha_slow / taua_slow  # slow adaptation increment
                tn = f_time[k] + tref      # refractory period
                spikes.append(f_time[k])   # store spike time

        # Convolve Spikes with Calcium Impulse Response Function
        conv = convolve_spikes(f_time=f_time, spike_trace=spikes, f_cirf=f_cirf)
        return conv

    # Set Bounding Values for Parameters
    t0 = clock.perf_counter()
    # lower_bounds = {'alpha_slow': -1.0,
    #                 'alpha_fast': -1.0,
    #                 'taua_fast': 0.001,
    #                 'taua_slow': 5.0,
    #                 'cirf_tau_1': 0.001,
    #                 'cirf_tau_2': 1.0,
    #                 'stimulus_intensity': 0.5,
    #                 'taum': 0.001}
    #
    # upper_bounds = {'alpha_slow': 1.0,
    #                 'alpha_fast': 1.0,
    #                 'taua_fast': 4.0,
    #                 'taua_slow': 20.0,
    #                 'cirf_tau_1': 2,
    #                 'cirf_tau_2': 15.0,
    #                 'stimulus_intensity': 5.0,
    #                 'taum': 0.2}

    lower_bounds = {'alpha_slow': -0.5,
                    'alpha_fast': -0.5,
                    'slow_dependency': -0.5}

    upper_bounds = {'alpha_slow': 1.0,
                    'alpha_fast': 1.0,
                    'slow_dependency': 2}
    initial_values = [-0.5, -0.5, 2]

    lower_bounds_list = []
    upper_bounds_list = []
    for lu, up in zip(lower_bounds, upper_bounds):
        lower_bounds_list.append(lower_bounds[lu])
        upper_bounds_list.append(upper_bounds[lu])

    # Set loss function
    # available loss functions: 'linear', 'soft_l1', 'huber', 'cauchy' and 'arctan'
    loss_function = 'huber'

    # Run non-linear least square fitting
    # out = curve_fit(lif_for_fitting, data_time, data, p0=initial_values, bounds=(lower_bounds_list, upper_bounds_list),
    #                 maxfev=15, full_output=True, loss=loss_function)
    out = curve_fit(lif_for_fitting, data_time, data, method='dogbox', p0=initial_values, maxfev=20, full_output=True, loss=loss_function)

    popt = out[0]
    pcov = out[1]
    nfev = out[2]['nfev']

    optimal_parameters = dict()
    for kk, vv in enumerate(lower_bounds):
        optimal_parameters[vv] = popt[kk]

    # confidence intervals (from t dist)
    alpha = 0.05  # 95% confidence interval = 100*(1-alpha)
    n = len(data)  # number of data points
    p = len(lower_bounds_list)  # number of parameters
    dof = max(0, n - p)  # number of degrees of freedom

    # student-t value for the dof and confidence level
    tval = t_stats.ppf(1.0 - alpha / 2., dof)
    t1 = clock.perf_counter()
    print('')
    print(f'Fitting took: {np.round((t1-t0)/60, 2)} mins with {nfev} function calls')
    print('')
    print(f'Fitting of the Neuron Model')
    print(f'---------------------------')
    print(f'Used loss function: {loss_function}')
    print('')
    for i, p, var in zip(range(n), optimal_parameters, np.diag(pcov)):
        sigma = var ** 0.5
        print(f'{p} : {optimal_parameters[p]:.3f} [{(optimal_parameters[p] - sigma * tval):.3f}  {(optimal_parameters[p] + sigma * tval):.3f}]')

    # Run lif model with optimal parameters
    model_results = lif_with_outputs(data_time, **optimal_parameters)
    model_results['loss_function'] = loss_function
    optimal_model_ca_trace = model_results['ca']
    recording_z = (data - np.mean(data)) / np.std(data)
    model_z = (optimal_model_ca_trace - np.mean(optimal_model_ca_trace)) / np.std(optimal_model_ca_trace)

    # Linear Model
    # Run Linear Model (on z score traces)
    lm_x = model_z
    lm_y = recording_z

    # start = 0
    # end = 1000
    # lm_x = lm_x[(tt >= start) & (tt < end)]
    # lm_y = lm_y[(tt >= start) & (tt < end)]
    lm_r2, lm_slope = linear_model(xx=lm_x, yy=lm_y, norm=True)
    sum_squares = np.sum((lm_x - lm_y) ** 2) / len(lm_x)
    # lr_results = {'r_squared': lm_r2, 'slope': lm_slope, 'ms': sum_squares}
    optimal_parameters['lrm_r_squared'] = lm_r2
    optimal_parameters['lrm_slope'] = lm_slope
    optimal_parameters['lrm_ms'] = sum_squares

    t2 = clock.perf_counter()
    print('')
    print('')
    print(f'Linear Regression took: {(t2-t1):.4f} secs')
    print(f'Linear Regression between optimal LIF Model and Ca-Recording')
    print(f'------------------------------------------------------------')
    print('')
    print(f'R²: {lm_r2:.4f}, slope: {lm_slope:.4f}, MS: {sum_squares:.4f}')
    print('')

    # Reconstruct optimal CIRF Function
    # dt = data_time[1] - data_time[0]  # time step
    # c_f_time = np.arange(0, optimal_parameters['cirf_tau_2'] * 5, dt)
    # c_f = create_cif_double_tau(tau1=optimal_parameters['cirf_tau_1'], tau2=optimal_parameters['cirf_tau_2'], dt=dt, a=1)

    fig, ax = plt.subplots(2, 1)
    # ax[0].plot(c_f_time, c_f, 'k')
    # ax[0].set_title(f"CIRF: tau1={optimal_parameters['cirf_tau_1']:.2f}, tau2={optimal_parameters['cirf_tau_2']:.2f}")
    ax[1].plot(data_time, recording_z, 'b')
    ax[1].plot(data_time, model_z, 'r')
    plt.show()
    return model_results, optimal_parameters


def down_sample_data(data, factor):
    factor = int(factor)
    data_down = data[::factor]
    return data_down


def down_sampling_results(time, stimulus, results, ca_recording, down_sampling_factor=10):
    # returns from model are
    #        {'spikes': np.asarray(spikes), 'volt': np.asarray(membrane_voltage),
    #         'a_slow': np.asarray(adaptation_slow), 'a_fast': np.asarray(adaptation_fast), 'ca': conv,
    #         'time_cirf': cirf_time, 'cirf': f_cirf, 'cirf_tau_1': cirf_tau_1, 'cirf_tau_2': cirf_tau_2}

    ca = down_sample_data(results['ca'], down_sampling_factor)
    a_slow = down_sample_data(results['a_slow'], down_sampling_factor)
    a_fast = down_sample_data(results['a_fast'], down_sampling_factor)
    stimulus_down = down_sample_data(stimulus, down_sampling_factor)
    time_down = down_sample_data(time, down_sampling_factor)
    ca_recording_down = down_sample_data(ca_recording, down_sampling_factor)
    model_results = pd.DataFrame(np.array([time_down, stimulus_down, ca, a_slow, a_fast, ca_recording_down]).T,
                                 columns=['time', 'stimulus', 'ca', 'a_slow', 'a_fast', 'recording'])

    spikes = pd.DataFrame(results['spikes'], columns=['spike_times'])
    return model_results, spikes


if __name__ == '__main__':
    recording_name = '220414_02_01'
    # base_dir = f'E:/CaImagingAnalysis/Paper_Data/Sound/{recording_name}/'
    base_dir = f'E:/CaImagingAnalysis/Paper_Data/Habituation/{recording_name}/'
    save_dir = f'E:/CaImagingAnalysis/Paper_Data/lif_model/fitting/'

    # Load Data
    time, stimulation = get_stimulus(stimulation_dir=f'{base_dir}{recording_name}_protocol.csv', stimulus_strength=1)
    f_raw_dir = f'{base_dir}{recording_name}_raw.txt'
    f_raw = pd.read_csv(f_raw_dir)

    # Compute delta f over f for recording trace
    df_f, f_rois, fbs = compute_df_over_f(f_raw, window_size=200, per=5, fast=True)
    roi = 'roi_26'
    ca_trace_recording = df_f[roi]
    fr_rec = 2.0345147125756804

    # Interpolation Ca Recording Trace for Linear Model
    time_recording = convert_samples_to_time(ca_trace_recording.to_numpy(), fr=fr_rec)
    f = interp1d(time_recording, ca_trace_recording, kind='linear', bounds_error=False, fill_value=0)
    ca_trace_recording = f(time)

    # Run non-linear least square fitting for lif model onto recording data
    results_model, settings = fit_model(time, ca_trace_recording)

    exit()
    # STORE DATA TO HDD
    # Down-Sampling
    results_model_down, spikes = down_sampling_results(time, stimulation, results_model,
                                                       ca_trace_recording, down_sampling_factor=10)
    results_model_down.to_csv(f'{save_dir}{recording_name}_{roi}_model_result.csv')
    spikes.to_csv(f'{save_dir}{recording_name}_{roi}_spikes.csv')
    np.save(f'{save_dir}{recording_name}_{roi}_optimal_settings.npy', settings)
    print('Optimal Model Data stored to HDD')

