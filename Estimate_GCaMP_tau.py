import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from tkinter import Tk
from IPython import embed
import pandas as pd
import analysis_util_functions as uf
import os


def create_cif(fr, tau):
    t_max = tau * 5  # in sec
    t_cif = np.arange(0, t_max*fr, 1)
    tau_samples = tau * fr
    cif = np.exp(-(t_cif/tau_samples))
    return cif


def fit_exp(x, y, full_exp, plot_results=False):
    # Fit the function a * np.exp(b * t) + c to x and y
    def func_full(xx, aa, x_tau, cc):
        return aa * np.exp(-(xx/x_tau)) + cc

    def func(xx, x_tau):
        return np.exp(-(xx/x_tau))

    if full_exp:
        popt, pcov = curve_fit(func_full, x, y)
        a = popt[0]
        b = popt[1]  # this is tau
        c = popt[2]
    else:
        popt, pcov = curve_fit(func, x, y)
        a = popt[0]  # this is tau
        b = 1
        c = 0
    # popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(-(b * t)) + c, x, y)

    if plot_results:
        # Create the fitted curve
        x_fitted = np.linspace(np.min(x), np.max(x), 100)
        # y_fitted = a * np.exp(-(b * x_fitted)) + c
        if full_exp:
            y_fitted = a * np.exp(-(x_fitted/b)) + c
        else:
            y_fitted = np.exp(-(x_fitted / a))
        # Plot
        ax = plt.axes()
        ax.scatter(x, y, label='Raw data')
        ax.plot(x_fitted, y_fitted, 'k', label='Fitted curve')
        ax.plot([0, np.max(x_fitted)], [0.36, 0.36], 'r')
        ax.set_title(r'Using curve\_fit() to fit an exponential function')
        ax.set_ylabel('y-Values')
        ax.set_xlabel('x-Values')
        ax.legend()
        plt.show()
    return popt, pcov


if __name__ == '__main__':
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    # SELECT TIF FILE
    uf.msg_box('INFO', 'PLEASE SELECT TIF FILE', '+')
    rec_file_path = askopenfilename()
    uf.msg_box('INFO', 'Please Wait ...', '+')

    # Load Raw F values (filtered)
    f_raw = pd.read_csv(rec_file_path, index_col=0)
    sig_data, f_rois, fbs = uf.compute_df_over_f(
        f_values=f_raw,
        window_size=0,
        per=5,
        fast=True)

    # Load stimulus protocol
    # go up one dir
    path_parent = os.path.dirname(os.path.dirname(rec_file_path))
    stimulation = uf.import_txt_stimulation_file(f'{path_parent}/stimulation/', 'stimulation', float_dec='.')
    # stimulation = pd.read_csv(f'{path_parent}/stimulation/stimulation.txt', index_col=0)

    # Cut out nice responses to use as data for fitting
    samples = []
    # -----
    print('Type in "-1" to stop collecting data')
    roi_nr = int(input('Enter ROI nr: '))
    while roi_nr != -1:
        roi = f'Mean{roi_nr}'
        s = uf.manual_data_selection(data=sig_data[roi], sort_samples=True)
        # Cut out data
        for k in range(s.shape[0]):
            d = sig_data[roi].iloc[s['Start'].iloc[k]:s['End'].iloc[k]]
            samples.append(d)
        roi_nr = int(input('Enter ROI nr: '))

    # Estimate tau by using exponential fitting
    fr_ca_rec = uf.estimate_sampling_rate(sig_data, stimulation, print_msg=True)
    # fr_ca_rec = 2
    fit_tau = []
    fit_tau_perr = []
    count_dismissed = 0
    for k_sample in samples:
        # d_t = uf.convert_samples_to_time(k_sample, fr_ca_rec)  # in sec
        y_data = k_sample / np.max(k_sample)
        d_t = np.arange(0, len(y_data), 1)
        fit_popt, fit_pcov = fit_exp(d_t, y_data, full_exp=False, plot_results=True)
        # one standard deviation error for parameters
        perr = np.sqrt(np.diag(fit_pcov))
        if len(perr) == 1:
            tau = fit_popt[0]
            err_tau = perr[0]
        else:
            tau = fit_popt[1]
            err_tau = perr[1]

        if err_tau >= 2:  # Error is in Standard Deviations away from estimated value
            count_dismissed += 1
        else:
            fit_tau_perr.append(err_tau)
            fit_tau.append(np.round(tau, 3))
    # Convert samples back to time
    estimated_taus = np.array(fit_tau) / fr_ca_rec
    estimated_mean_tau = np.mean(estimated_taus)
    estimate_sd_tau = np.std(estimated_taus)
    print(f'Estimated Average Tau Value: {np.round(estimated_mean_tau, 3)} +- {np.round(estimate_sd_tau, 3)}')
    embed()
    exit()