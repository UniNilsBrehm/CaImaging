import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
import pandas as pd
import time as clock


def convolve_spikes(f_time, spike_trace, f_cirf):
    f_spikes_binary = np.zeros(len(f_time))
    for f_spk in spike_trace:
        # Compute Spikes Binary
        f_spikes_binary[(f_time == f_spk)] = 1
        # Compute Calcium Fluorescence Trace
    f_cat = np.convolve(f_spikes_binary, f_cirf, 'full')[:len(f_spikes_binary)]
    return f_cat


def create_cif(tau, dt):
    t_max = tau * 5  # in sec
    t_cif = np.arange(0, t_max, dt)
    cif = np.exp(-(t_cif/tau))
    return cif


def create_cif_double_tau(tau1, tau2, a, dt):
    t_max = tau2 * 5  # in sec
    t_cif = np.arange(0, t_max, dt)
    cif = a * (1 - np.exp(-(t_cif/tau1)))*np.exp(-(t_cif/tau2))
    return cif


def spike_frequency(time, spikes, fill=0.0):
    zrate = 0.0 if fill == 'extend' else fill     # firing rate for empty trials
    rates = np.zeros((len(time), len(spikes)))
    for k in range(len(spikes)):                  # loop through trials
        isis = np.diff(spikes[k])                 # compute interspike intervals
        if len(spikes[k]) > 2:
            # interpolate inverse ISIs at `time`:
            fv = (1.0/isis[0], 1.0/isis[-1]) if fill == 'extend' else (fill, fill)
            fr = interp1d(spikes[k][:-1], 1.0/isis, kind='previous',
                          bounds_error=False, fill_value=fv)
            rate = fr(time)
        else:
            rate = np.zeros(len(time)) + zrate
        rates[:,k] = rate
    frate = np.mean(rates, 1)                     # average over trials
    return frate


def eif(time, stimulus, vthresh=1.0, tref=0.003, taum=0.01, delta=1.0, vreset=0.0, noisedv=0.01, vrh=1.0, noiseda=0.001,
        taua=0.01, b=0.1, a=0.1):
    # a: sub-threshold adaptation
    # b: spike-triggered adaptation
    tn = time[0]
    dt = time[1] - time[0]
    noisev = np.random.randn(len(stimulus)) * noisedv / np.sqrt(dt)  # properly scaled voltage noise term
    noisea = np.random.randn(len(stimulus))*noiseda/np.sqrt(dt)  # properly scaled adaptation noise term
    V = np.random.rand()*(vthresh-vreset) + vreset
    A = 0
    membrane_voltage = []
    spikes = []
    adaptation = []
    for kk, _ in enumerate(stimulus):
        membrane_voltage.append(V)  # store membrane voltage value
        adaptation.append(A)
        if time[kk] < tn:
            continue                 # no integration during refractory period
        V += (-V + delta * np.exp((V-vrh) / delta) + stimulus[kk] + noisev[kk] -A) * dt/taum
        A += (-A + noisea[kk] + a*(V-vreset)) * dt / taua  # adaptation dynamics

        if V > vthresh:              # threshold condition
            V = vreset               # voltage reset
            A += b / taua
            tn = time[kk] + tref      # refractory period
            spikes.append(time[kk])   # store spike time
    return np.asarray(spikes), np.asarray(membrane_voltage), np.asarray(adaptation)


def lifac(time, stimulus, f_cirf, taum=0.01, tref=0.003, noisedv=0.01,
          vreset=0.0, vthresh=1.0, noiseda=0.01, taua_fast=0.01, alpha_fast=0.1,
          rng=np.random, alpha_slow=0.1, taua_slow=0.5, slow_dependency=0.5):
    dt = time[1] - time[0]                                # time step
    noisev = rng.randn(len(stimulus))*noisedv/np.sqrt(dt) # properly scaled voltage noise term
    noisea = rng.randn(len(stimulus))*noiseda/np.sqrt(dt) # properly scaled adaptation noise term

    # initialization:
    # np.seterr('raise')
    tn = time[0]
    V = rng.rand()*(vthresh-vreset) + vreset
    A = 0.0
    A_fast = 0.0
    A_slow = 0.0

    stimulus_diff = np.diff(stimulus, append=0)
    # integration:
    spikes = []
    membrane_voltage = []
    adapation_slow = []
    adapation_stimulus = []
    adapation_fast = []
    for k in range(len(stimulus)):
        membrane_voltage.append(V)  # store membrane voltage value
        adapation_stimulus.append(A)  # store adaptation value
        adapation_fast.append(A_fast)
        adapation_slow.append(A_slow)
        if time[k] < tn:
            continue                 # no integration during refractory period
        # V += (-V - A - A_fast + A_slow + stimulus[k] + noisev[k])*dt/taum   # membrane equation
        V += (-V - A - A_fast - A_slow + stimulus[k] + noisev[k])*dt/taum   # membrane equation
        # A += (-A + noisea[k])*dt/taua                 # stimulus offset adaptation dynamics
        A_fast += (-A_fast + noisea[k])*dt/taua_fast  # fast adaptation dynamics
        A_slow += (-A_slow + noisea[k])*dt/taua_slow  # slow adaptation dynamics

        # if stimulus_diff[k] < 0:
            # A += (stimulus[k] * alpha + alpha) / taua  # stimulus offset adaptation increment

        if V > vthresh:              # threshold condition
            V = vreset               # voltage reset
            A_fast += alpha_fast/taua_fast                                       # fast adaptation increment
            # A_slow += np.round(slow_dependency*A_slow, 2) * alpha_slow/taua_slow   # slow adaptation increment

            if (A_slow > 0.001) & (slow_dependency > 0):
                A_slow += A_slow * slow_dependency  # slow adaptation increment
            else:
                A_slow += alpha_slow/taua_slow   # slow adaptation increment

            tn = time[k] + tref      # refractory period
            spikes.append(time[k])   # store spike time

    # Convolve Spikes with Calcium Impulse Response Function
    conv = convolve_spikes(f_time=time, spike_trace=spikes, f_cirf=f_cirf)
    return {'spikes': np.asarray(spikes), 'volt': np.asarray(membrane_voltage),
            'a_slow': np.asarray(adapation_slow), 'a_fast':  np.asarray(adapation_fast), 'ca': conv}


def plot_model(ax, time, stimulus, spikes, ca_trace, v, a, a_fast):
    if len(spikes[0]) == 0:
        print('ERROR: NO SPIKES')
        exit()
    ca_trace = (ca_trace / np.max(ca_trace)) * np.max(stimulus)
    ax[0].plot(time, stimulus, 'k', alpha=0.5)
    ax[0].fill_between(time, stimulus, facecolor='g', alpha=0.2)

    # ax[0].plot(spikes, [1] * len(spikes), 'sk')
    # ax[0].plot(time, v[0], 'b', lw=0.5)
    ax[0].eventplot(spikes[0], colors=['k'], lineoffsets=np.max(stimulus) + 0.5, lw=1)
    ax[0].plot(time, a[0], 'tab:red', lw=2, alpha=1)
    ax[0].plot(time, a_fast[0], 'tab:orange', lw=2, alpha=1)
    ax[0].plot(time, a_fast[0]+a[0], 'tab:cyan', lw=2, alpha=0.5)

    ax[1].plot(time, stimulus, 'k', alpha=0.5)
    ax[1].fill_between(time, stimulus, facecolor='g', alpha=0.2)
    ax[1].plot(time, ca_trace, 'g')
    # ax[1].eventplot(spikes[0], colors=['k'], lineoffsets=0, lw=2)

    # a new time array with less temporal resolution than the original one:
    ratetime = np.arange(time[0], time[-1], 0.01)
    frate = spike_frequency(ratetime, spikes, 0.1)
    ax2 = ax[2].twinx()
    ax2.eventplot(spikes, colors=['k'], lineoffsets=np.arange(1, len(spikes)+1), lw=0.4, alpha=0.75)
    ax[2].plot(ratetime, frate, 'r', lw=2)
    ax2.set_yticks([])
    ax[0].set_xlim(-0.5, time[-1])
    ax[1].set_xlim(-0.5, time[-1])
    ax[2].set_xlim(-0.5, time[-1])
    ax[0].set_ylim(0, np.max(stimulus) + 1)
    ax[1].set_ylim(0, np.max(ca_trace) + 1)
    ax[2].set_ylim(0, len(spikes) + 1)
    if np.max(frate) >= len(spikes):
        ax[2].set_ylim(0, np.max(frate) + 1)
        ax[2].fill_between(time, stimulus / np.max(stimulus) * np.max(frate) + 1, facecolor='g', alpha=0.2)
    else:
        ax[2].fill_between(time, stimulus / np.max(stimulus) * len(spikes) + 1, facecolor='g', alpha=0.2)


def down_sample(data, factor):
    factor = int(factor)
    data_down = data[::factor]
    return data_down


def stimulus_pulse_train(dt=0.001):
    # dt: integration time step in seconds (should be dt <= 10xmembrane tau)
    before = np.zeros(len(np.arange(0.0, 5, dt)))
    after = np.zeros(len(np.arange(0.0, 5, dt)))
    n = 10              # number of pulses
    T = 2         # period of the pulses in seconds
    t0 = 0          # start of the pulse within the period in seconds
    t1 = 1         # end of the pulse within the period in seconds
    time = np.arange(0.0, n*T, dt)
    stimulus = np.zeros(len(time))
    stimulus[(time%T>t0) & (time%T<t1)] = 2
    stimulus = np.concatenate([before, stimulus, after])
    time = np.arange(0.0, n*T + 10, dt)
    return time, stimulus


def stimulus_single_pulses(pulse_times, pulse_dur, pulse_intensity=1, time_max=30, dt=0.001):
    # dt: integration time step in seconds (should be dt <= 10xmembrane tau)
    if len(pulse_times) != len(pulse_dur):
        print('ERROR: pulse_times and pulse_dur must have same size!')
    time = np.arange(-0.2, time_max+dt, dt)
    stimulus = np.zeros(len(time)) + 0
    for pulse_t, pulse_d in zip(pulse_times, pulse_dur):
        stimulus[(time > pulse_t) & (time < pulse_t+pulse_d)] = pulse_intensity
    return time, stimulus


# ----------------------------------------------------------------------------------------------------------------------
# neuron_type = 'sensitizing'
neuron_type = input('Enter Neuron Type (non_adapting, adapting, sensitizing): ')
print('')

recording_name = '220414_02_01'
stimulation_dir = f'E:/CaImagingAnalysis/Paper_Data/Habituation/{recording_name}/{recording_name}_protocol.csv'
# stimulation_dir = f'E:/CaImagingAnalysis/Paper_Data/Sound/Duration/220520_DOB/{recording_name}/{recording_name}_protocol.csv'
# stimulation_dir = f'E:/CaImagingAnalysis/Paper_Data/Sound/Habituation/220520_DOB/{recording_name}/{recording_name}_protocol.csv'

stimulus_strength = 2

# Calcium Impulse Response Function Settings
# cirf_tau_1 = 0.04
# cirf_tau_2 = 4
cirf_tau_1 = 0.02
cirf_tau_2 = 3

# Set Leaky integrate-and-fire settings
if neuron_type == 'non_adapting':
    print(f'Selected: {neuron_type}')
    stimulus_strength = 2
    # Non Adapting Neuron
    model_settings = dict(taum=0.01, vthresh=1, noisedv=0.0001, noiseda=0.0001,
                          alpha_fast=0, taua_fast=0.9,
                          alpha_slow=0, taua_slow=12, slow_dependency=0.5)
elif neuron_type == 'adapting':
    print(f'Selected: {neuron_type}')
    stimulus_strength = 2
    # Adapting Neuron
    model_settings = dict(taum=0.01, vthresh=1, noisedv=0.0001, noiseda=0.0001,
                          alpha_fast=0.02, taua_fast=1.0,
                          alpha_slow=0.02, taua_slow=18, slow_dependency=0.05)
elif neuron_type == 'sensitizing':
    print(f'Selected: {neuron_type}')
    stimulus_strength = 1.0
    # Sensitizing Neuron
    model_settings = dict(taum=0.01, vthresh=1, noisedv=0.0001, noiseda=0.0001,
                          alpha_fast=-0.005, taua_fast=0.5,
                          alpha_slow=-0.04, taua_slow=10, slow_dependency=0)
else:
    print(f'Selected: {neuron_type}')
    print('Invalid Neuron Type. Set to default')
    model_settings = dict(taum=0.01, vthresh=1, noisedv=0.0001, noiseda=0.0001,
                          alpha_fast=0, taua_fast=0.9,
                          alpha_slow=0, taua_slow=12, slow_dependency=0.5)

# Load Protocol
protocol = pd.read_csv(stimulation_dir, index_col=0)

dt = 0.001
time_max = np.round(protocol['Offset_Time'].max() + 10)
time = np.arange(-0.2, time_max + dt, dt)
stimulus = np.zeros(len(time)) + 0
for t_on, t_off in zip(protocol['Onset_Time'], protocol['Offset_Time']):
    stimulus[(time >= t_on) & (time < t_off)] = stimulus_strength


# Create stimulus
# time, stimulus = stimulus_single_pulses([5, 15], [5, 5], pulse_intensity=10, time_max=30, dt=0.001)

# Stimulus-specific adaptation
# time, stimulus = stimulus_pulse_train(dt=0.001)

# Compute Calcium Impulse Response Function
# cirf_tau_1 = 0.01
# cirf_tau_2 = 1.8

cirf_time = np.arange(0, cirf_tau_2*5, dt)
cirf = create_cif_double_tau(tau1=cirf_tau_1, tau2=cirf_tau_2, dt=dt, a=1)

# Compute Model Output
results = lifac(time, stimulus, cirf, **model_settings)
# results = Parallel(n_jobs=-1)(delayed(lifac)(time, stimulus, cirf, **model_settings) for i in range(ntrials))

# Down Sampling
down_sampling_factor = 10
ca = down_sample(results['ca'], down_sampling_factor)
a_slow = down_sample(results['a_slow'], down_sampling_factor)
a_fast = down_sample(results['a_fast'], down_sampling_factor)
stimulus_down = down_sample(stimulus, down_sampling_factor)
time_down = down_sample(time, down_sampling_factor)

model_results = pd.DataFrame(np.array([time_down, stimulus_down, ca, a_slow, a_fast]).T,
                             columns=['time', 'stimulus', 'ca', 'a_slow', 'a_fast'])

spikes = pd.DataFrame(results['spikes'], columns=['spike_times'])

print('Model Computation Done')

# Collect Parallel Data
# spikes = pd.DataFrame()
# model_results = pd.DataFrame()
# for kk, vv in enumerate(results):
#     spikes = pd.concat([spikes, pd.DataFrame(np.asarray([vv['spikes'], np.zeros(len(vv['spikes'])) + kk + 1]).T,
#                                              columns=['spikes', 'trials'])])
#
#     model_results = pd.concat(
#         [model_results,
#          pd.DataFrame(np.asarray([vv['ca'], vv['a_slow'], vv['a_fast'], vv['volt'], np.zeros(len(vv['a_slow'])) + kk + 1]).T,
#                       columns=['ca', 'a_slow', 'a_fast', 'volt', 'trials'])]
#     )

# Store Results to HDD
save_dir = f'E:/CaImagingAnalysis/Paper_Data/lif_model/'
model_results.to_csv(f'{save_dir}{recording_name}_{neuron_type}_model_result.csv')
spikes.to_csv(f'{save_dir}{recording_name}_{neuron_type}_spikes.csv')
settings = {
           'settings_a': model_settings,
           'cirf': cirf, 'cirf_tau_01': cirf_tau_1, 'cirf_tau_02': cirf_tau_2, 'cirf_time': cirf_time
           }
np.save(f'{save_dir}{recording_name}_{neuron_type}_settings.npy', settings)

print('Model Results stored to HDD')

