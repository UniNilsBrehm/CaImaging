import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from scipy.interpolate import interp1d
from joblib import Parallel, delayed


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


def lifac(time, stimulus, taum=0.01, tref=0.003, noisedv=0.01,
          vreset=0.0, vthresh=1.0, taua=0.1, alpha=0.05, noiseda=0.01, rng=np.random, adapt=True):
    dt = time[1] - time[0]                                # time step
    noisev = rng.randn(len(stimulus))*noisedv/np.sqrt(dt) # properly scaled voltage noise term
    noisea = rng.randn(len(stimulus))*noiseda/np.sqrt(dt) # properly scaled adaptation noise term
    # initialization:
    tn = time[0]
    V = rng.rand()*(vthresh-vreset) + vreset
    A = 0.0
    # integration:
    spikes = []
    membrane_voltage = []
    adapation_values = []
    for k in range(len(stimulus)):
        membrane_voltage.append(V)  # store membrane voltage value
        adapation_values.append(A)  # store adaptation value
        if time[k] < tn:
            continue                 # no integration during refractory period
        V += (-V - A + stimulus[k] + noisev[k])*dt/taum   # membrane equation
        if adapt:
            A += (-A + noisea[k])*dt/taua                     # adaptation dynamics
        if V > vthresh:              # threshold condition
            V = vreset               # voltage reset
            if adapt:
                A += alpha/taua          # adaptation increment
            tn = time[k] + tref      # refractory period
            spikes.append(time[k])   # store spike time

    return np.asarray(spikes), np.asarray(membrane_voltage), np.asarray(adapation_values)


def plot_model(ax, time, stimulus, spikes, ca_trace, v, a):
    if len(spikes[0]) == 0:
        print('ERROR: NO SPIKES')
        exit()
    ca_trace = (ca_trace / np.max(ca_trace)) * np.max(stimulus)
    ax[0].plot(time, stimulus, 'k', alpha=0.5)
    # ax[0].plot(spikes, [1] * len(spikes), 'sk')
    ax[0].plot(time, v[0], 'b')
    ax[0].plot(time, a[0], 'r')
    ax[0].eventplot(spikes[0], colors=['k'], lineoffsets=0, lw=2)

    ax[1].plot(time, stimulus, 'k', alpha=0.5)
    ax[1].plot(time, ca_trace, 'g')
    ax[1].eventplot(spikes[0], colors=['k'], lineoffsets=0, lw=2)

    # a new time array with less temporal resolution than the original one:
    ratetime = np.arange(time[0], time[-1], 0.1)
    frate = spike_frequency(ratetime, spikes, 0.1)
    ax2 = ax[2].twinx()
    ax2.eventplot(spikes, colors=['k'], lineoffsets=np.arange(1, len(spikes)+1), lw=0.4, alpha=0.2)
    ax[2].plot(ratetime, frate)                   # time axis in milliseconds

    ax2.set_yticks([])
    ax[0].set_xlim(-0.5, time[-1])
    ax[1].set_xlim(-0.5, time[-1])
    ax[2].set_xlim(-0.5, time[-1])
    ax[0].set_ylim(0, np.max(stimulus) + 1)
    ax[1].set_ylim(0, np.max(ca_trace) + 1)
    ax[2].set_ylim(0, len(spikes) + 1)
    if np.max(frate) >= len(spikes):
        ax[2].set_ylim(0, np.max(frate) + 1)


# Set Leaky integrate-and-fire settings
model_settings_adapt = dict(taum=0.01, taua=70, adapt=True,  vthresh=2.9, alpha=0.35, noisedv=0.25, noiseda=0.5)
model_settings_no_adapt = dict(taum=0.01, taua=4, adapt=False,  vthresh=3, alpha=1.2, noisedv=0.25, noiseda=0.01)

# Create stimulus
# dt = 0.001  # integration time step in seconds (should be dt <= 10xmembrane tau)
# time_max = 20
# time = np.arange(-0.2, time_max+dt, dt)
# stimulus = np.zeros(len(time)) + 0
# stimulus[(time > 1) & (time < 6)] = 4.0

# Stimulus-specific adaptation
dt = 0.001        # integration time step in seconds
before = np.zeros(len(np.arange(0.0, 5, dt)))
after = np.zeros(len(np.arange(0.0, 5, dt)))
n = 10              # number of pulses
T = 7         # period of the pulses in seconds
t0 = 0          # start of the pulse within the period in seconds
t1 = 2         # end of the pulse within the period in seconds
time = np.arange(0.0, n*T, dt)
stimulus = np.zeros(len(time))
stimulus[(time%T>t0) & (time%T<t1)] = 3.0
stimulus = np.concatenate([before, stimulus, after])
time = np.arange(0.0, n*T + 10, dt)

# Compute responses (trials)
spikes = []
voltage = []
adaptation = []
ntrials = 20
for k in range(ntrials):
    sp, v, a = lifac(time, stimulus, **model_settings_adapt)
    spikes.append(sp)
    voltage.append(v)
    adaptation.append(a)

spikes2 = []
voltage2 = []
adaptation2 = []
ntrials2 = 20
for k in range(ntrials2):
    sp, v, a = lifac(time, stimulus, **model_settings_no_adapt)
    spikes2.append(sp)
    voltage2.append(v)
    adaptation2.append(a)
# --------------------------------------------------------------------
spikes_binary = np.zeros(len(time))
for spk in spikes[0]:
    spikes_binary[(time == spk)] = 1

spikes_binary2 = np.zeros(len(time))
for spk in spikes2[0]:
    spikes_binary2[(time == spk)] = 1

# Compute Calcium Impulse Response Function
cirf_tau_1 = 0.1
cirf_tau_2 = 8
fr = len(voltage[0]) / time[-1]
cirf_time = np.arange(0, cirf_tau_2*5, dt)
cirf = create_cif_double_tau(tau1=cirf_tau_1, tau2=cirf_tau_2, dt=dt, a=1)

# Compute Calcium Fluorescence Trace
ca_trace = np.convolve(spikes_binary, cirf, 'full')
ca_trace = ca_trace[:len(spikes_binary)]
ca_trace2 = np.convolve(spikes_binary2, cirf, 'full')
ca_trace2 = ca_trace2[:len(spikes_binary2)]

# Plot
fig, axs = plt.subplots(3, 2, sharex=True)
plot_model(ax=axs[:, 0], time=time, stimulus=stimulus, spikes=spikes, ca_trace=ca_trace, v=voltage, a=adaptation)
plot_model(ax=axs[:, 1], time=time, stimulus=stimulus, spikes=spikes2, ca_trace=ca_trace2, v=voltage2, a=adaptation2)
axs[2, 0].set_xlabel('Time [s]')
axs[2, 1].set_xlabel('Time [s]')
axs[0, 0].set_ylabel('Membrane Voltage [mV]')
axs[1, 0].set_ylabel('dF/F')
axs[2, 0].set_ylabel('Firing Rate [Hz]')

# fig2, axs2 = plt.subplots()
# axs2.plot(cirf_time, cirf)
plt.show()
