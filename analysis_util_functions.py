import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from IPython import embed
from math import factorial
import more_itertools
import sys
from matplotlib.widgets import Button
from matplotlib.widgets import CheckButtons
from matplotlib.widgets import Slider
import time
from mpl_point_clicker import clicker
from tkinter.filedialog import askdirectory
from tkinter.filedialog import askopenfilename
from tkinter import Tk
from tkinter import messagebox
import cv2
import shutil
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit


def remove_axis(f_ax, x_ticks=True, y_ticks=True, x_line=True, y_line=True, box=False):
    # f_ax.set_xticks([], minor=True)
    # f_ax.set_xticks([], minor=True)
    # f_ax.set_axis_off()
    # f_ax.set_frame_on(False)
    # f_ax.get_xaxis().tick_bottom()
    if not box:
        f_ax.spines.top.set_visible(False)
        f_ax.spines.right.set_visible(False)
    if not y_line:
        f_ax.spines.left.set_visible(False)
    if not x_line:
        f_ax.spines.bottom.set_visible(False)
    if not x_ticks:
        f_ax.axes.get_xaxis().set_visible(False)
    if not y_ticks:
        f_ax.axes.get_yaxis().set_visible(False)


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


def import_csv_file(data_path, data_name):
    """ imports any csv file with comma separation """
    if data_path.endswith('/'):
        out = pd.read_csv(f'{data_path}{data_name}.csv', index_col=0)
    else:
        out = pd.read_csv(f'{data_path}/{data_name}.csv', index_col=0)
    return out


def import_txt_stimulation_file(data_path, data_name, float_dec='.', sep_symbol='\s+'):
    """ imports stimulation text file from MOM 2P Imaging Setup """

    if not data_name.endswith('.txt'):
        data_name = f'{data_name}.txt'
    if data_path.endswith('/'):
        out = pd.read_csv(f'{data_path}{data_name}', sep=sep_symbol, decimal=float_dec, header=None, names=['Time', 'Volt'])
    else:
        out = pd.read_csv(f'{data_path}/{data_name}', sep=sep_symbol, decimal=float_dec, header=None, names=['Time', 'Volt'])
    return out


def check_files(file_name, file_path):
    # Check if needed files are there:
    if file_name in os.listdir(file_path):
        print('')
        print('---------- INFO ----------')
        print(f'FOUND {file_name} FILE!')
        print('')
        return True
    else:
        print('')
        print('---------- WARNING ----------')
        print(f'COULD NOT {file_name} FILE')
        print('MAKE SURE IT IS NAMED CORRECTLY:')
        print(file_name)
        print('')
        return False


def estimate_sampling_rate(data, f_stimulation, print_msg):
    r""" Estimate the sampling rate via the total duration and sample count
        ----------
        data : pandas data frame, shape (N,)
            the values of all ROIs.
        f_stimulation : pandas data frame, shape (N,)
            stimulation recording (voltage trace and time trace).
        Returns
        -------
        fr : float
            the estimated sampling rate.
        Notes
        -----
        the stimulation data frame needs a column called 'Time' with sample time points.
    """
    if (type(data) is int) or (type(data) is float):
        data_size = data
    else:
        data_size = len(data)
    max_time = f_stimulation['Time'].max()
    fr = data_size / max_time
    if print_msg:
        print('')
        print('--------- INFO ---------')
        print(f'Estimated Frame Rate of Ca Imaging Recording: {fr} Hz')
        print('')

    return fr


def check_matching_durations(d1, d2, fr1, acceptable_dur_diff):
    # INPUTS:
    # d1: fluorescence values in samples (pandas data frame)
    # d2: stimulation file (pandas data frame)
    t_f = convert_samples_to_time(d1, fr1)[-1]
    t_s = d2['Time'].iloc[-1]
    dur_diff = abs(t_s - t_f)
    if dur_diff <= acceptable_dur_diff:
        print('')
        print('--------- INFO ---------')
        print(f'Recording and Stimulus have the same length! -- Diff. below {acceptable_dur_diff} s')
        print(f'Difference in duration: {dur_diff} s')
        print('')
        return True
    else:
        print('')
        print('---------- WARNING ----------')
        print('Recording and Stimulus DONT have the same length!')
        print(f'Ca Imaging Recording Duration: {t_f} s')
        print(f'Stimulus Duration: {t_s} s')
        print(f'Difference in duration: {dur_diff} s')
        print('')
        return False


def plot_stimulus_response_trace(sig, sig_raw, sig_t, st, st_t, st_on, st_off, show):
    y_lim = 4
    fig, ax = plt.subplots(2, 1, sharex=True)

    # Plot Stimulus Trace
    ax[0].plot(st_t, st, 'k')
    ax[0].set_ylabel('Voltage [V]')
    remove_axis(ax[0], x_ticks=False, x_line=False)

    # Plot Response Traces
    ax[1].plot(sig_t, sig_raw, alpha=0.5, lw=1.5, color='k')
    # filtered = savitzky_golay(y=sig.to_numpy(), window_size=31, order=1)
    ax[1].plot(sig_t, sig, color='k')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('dF/F')
    ax[1].set_ylim([-1, y_lim])

    # plot stimulus marker
    marker_color = 'g'
    for k in range(len(st_on)):
        ax[1].fill_between([st_on[k], st_off[k]], [y_lim, y_lim], color=marker_color,  edgecolor=marker_color)
    remove_axis(ax[1], box=False)

    # Settings
    fig.subplots_adjust(left=0.2, top=0.9, bottom=0.2, right=0.9, wspace=0.2, hspace=0.05)
    cm = 1 / 2.54  # centimeters in inches
    # width, height
    fig.set_size_inches(8 * cm, 10 * cm)

    if show:
        plt.show()
    else:
        return fig, ax


def check_directory_structure(dir_path, dir_control):
    dir_control = sorted(dir_control)
    dir_list = sorted(os.listdir(dir_path))
    if dir_list == dir_control:
        print('')
        print('-------- INFO --------')
        print('Selected Directory is valid! Program will start soon...')
        print('')

        return True
    else:
        print('')
        print('-------- WARNING --------')
        print('Selected Directory is invalid! Please make sure the needed sub-folders and files are present!')
        print(dir_control)
        print('')
        return False


def data_viewer(rec_name, f_raw, sig_t, ref_im, st_rec, protocol, rois_dic, good_cells_list, scores):
    """ fraw: raw fluorescence values"""

    print('')
    print('---------- STARTING DATA VIEWER ----------')
    print('')

    # sig_data, f_rois, fbs = compute_df_over_f(
    #     f_values=filter_raw_data(f_raw, win=0, o=1),
    #     window_size=0,
    #     per=5,
    #     fast=True)

    # Get Stimulus Info
    st_on = protocol['Onset_Time']
    st_off = protocol['Offset_Time']

    # Get Roi Names
    f_rois_all = f_raw.keys()
    # Normalize raw values to max = 1
    f_raw_norm = f_raw / f_raw.max()
    # y_lim = np.max(sig_data.max())
    y_lim = 1.5
    y_lim_min = -0.1

    reg_scores = scores
    good_c = good_cells_list

    # Select first data to plot: ROI 1
    # sig = sig_data[f_rois[0]]
    sig = f_raw_norm[f_rois_all[0]]

    # Create the initial figure ----------
    fig, ax = plt.subplots(2, 1)
    # Plot Ref Image
    ref_im_x = ref_im.shape[0]
    ref_im_y = ref_im.shape[1]
    # re_im_obj = ax[0].imshow(ref_im, extent=[0, ref_im_y, 0, ref_im_x], aspect='equal')
    re_im_obj = ax[0].imshow(ref_im)
    ax[0].axis('off')  # clear x-axis and y-axis

    # plot stimulus marker
    ax[1].plot(st_rec['Time'], (st_rec['Volt'] / st_rec['Volt'].max()) * y_lim, color='darkgreen', lw=2, alpha=0.3)
    marker_color = 'g'
    stim_text_obj = {}
    for k in range(len(st_on)):
        ax[1].fill_between([st_on[k], st_off[k]], [y_lim, y_lim], color=marker_color,  edgecolor=marker_color, alpha=0.3)
        stim_text_1 = protocol['Stimulus'].iloc[k]
        stim_text_2 = protocol['Duration'].iloc[k]
        stim_text_obj[k] = ax[1].text(
            protocol['Onset_Time'].iloc[k], 1, f'{stim_text_1[0]}\n{np.round(stim_text_2/1000, 2)}',
            fontsize=6, color=(0, 0, 0), horizontalalignment='left', verticalalignment='center'
        )
    remove_axis(ax[1], box=False)
    # Plot Stimulus Trace
    ax[0].set_title(f'ROI: {1}')

    # protocol labels


    # Plot Response Traces
    l, = ax[1].plot(sig_t, sig, color='k')
    ax[1].set_xlabel('Time [s]')
    # ax[1].set_ylabel('dF/F')
    ax[1].set_ylabel('Raw Values')
    ax[1].set_ylim([y_lim_min, y_lim])

    text_obj = ax[0].text(
        0, 0, '', fontsize=12, color=(1, 0, 0),
        horizontalalignment='center', verticalalignment='center'
    )

    text_score_obj_x = -(np.max(sig_t) // 4)
    text_score_obj_y = 0

    text_score_obj = ax[1].text(
        text_score_obj_x, text_score_obj_y, 'scores: ', fontsize=12, color=(0, 0, 0),
        horizontalalignment='center', verticalalignment='center'
    )

    # Set Window to Full Screen
    manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    manager.window.showMaximized()
    fig.subplots_adjust(left=0.2, top=0.9, bottom=0.2, right=0.9, wspace=0.1, hspace=0.25)

    class Index:
        # Initialize parameter values

        th_reg = 0.2
        ind = 0
        df_win = 0
        df_per = 0
        filter_win = 0
        filter_order = 3
        show_raw = True
        # data, f_rois, fbs = compute_df_over_f(f_values=f_raw, window_size=df_win, per=df_per, fast=True)
        data = f_raw_norm
        y_lim = np.max(data.max())
        f_rois = f_rois_all
        selected_roi = f_rois[0]
        cell_color = 'red'
        good_cells_list = good_c
        if len(good_c) <= 1:
            print('Could not find a good cell list ... Creating new one ...')
            good_cells_list = np.zeros(len(f_rois))
        # Compute Ref Image
        active_roi_nr = 1
        roi_images = []
        roi_pos = []
        for ii, active_roi in enumerate(rois_dic):
            roi_pos.append((rois_dic[active_roi]['left'] + rois_dic[active_roi]['width'] // 2,
                            rois_dic[active_roi]['top'] + rois_dic[active_roi]['height'] // 2))
            roi_images.append(draw_rois_zipped(
                img=ref_im, rois_dic=rois_dic, active_roi=rois_dic[active_roi],
                r_color=(0, 0, 255), alp=0.5, thickness=1)
            )

        def turn_page(self):
            # Select new ROI Number
            self.selected_roi = self.f_rois[self.ind]

            # Check Cell Button
            self.flip_button(status=self.good_cells_list.iloc[int(self.selected_roi) - 1, 0])

            # Update Data Display
            ydata = self.data[self.selected_roi]
            l.set_ydata(ydata)

            # Update Ref Image
            re_im_obj.set_data(self.roi_images[int(self.selected_roi) - 1])

            # Update Ref Image Label
            text_obj.set_text(f'{self.selected_roi}')
            text_obj.set_position(self.roi_pos[int(self.selected_roi) - 1])

            # Update Score Text
            for kk, vv in enumerate(reg_scores[self.selected_roi]):
                if vv >= self.th_reg:
                    stim_text_obj[kk].set_color((1, 0, 0))
                else:
                    stim_text_obj[kk].set_color((0, 0, 0))

            if any(reg_scores[self.selected_roi] >= self.th_reg):
                text_score_obj.set_color((1, 0, 0))
            else:
                text_score_obj.set_color((0, 0, 0))
            score_text = reg_scores[['type', 'parameter', self.selected_roi]].round(2).to_string(header=None,
                                                                                                 index=None)
            text_score_obj.set_text(f'Reg. Scores \n{score_text}')

            # Set the line color of the last roi to red, just for visualization
            if self.ind + 1 == len(self.f_rois):
                l.set_color('red')
            else:
                l.set_color('black')
            plt.draw()

        def next(self, event):
            # Increase Index by 1
            self.ind += 1
            self.ind = self.ind % self.data.shape[1]
            self.turn_page()

        def prev(self, event):
            # Decrease Index by 1
            self.ind -= 1
            self.ind = self.ind % self.data.shape[1]
            self.turn_page()

        def update_fbs_win_slider(self, val):
            # Get new parameter value
            self.df_win = int(val)
            # Compute data based on new parameter value
            self.compute_data()
            # Update y-lim
            self.y_lim = np.max(self.data.max())
            ax[1].set_ylim([y_lim_min, self.y_lim])
            l.set_ydata(self.data[self.f_rois[self.ind]])
            plt.draw()

        def update_fbs_per_slider(self, val):
            self.df_per = int(val)
            # Compute data based on new parameter value
            self.compute_data()
            # Update y-lim
            self.y_lim = np.max(self.data.max())
            ax[1].set_ylim([y_lim_min, self.y_lim])
            l.set_ydata(self.data[self.f_rois[self.ind]])
            plt.draw()

        def update_filter_win_slider(self, val):
            self.filter_win = val
            # Check if order > window size - 1
            if self.filter_win > 0:  # Filter is ON
                if self.filter_order >= self.filter_win - 1:
                    print('')
                    print('---------- WARNING ----------')
                    print(f'Polynomial order to big for window size: {self.filter_win}')
                    self.filter_order = self.filter_win - 2
                    if self.filter_order <= 0:
                        self.filter_order = 1
                    print(f'Polynomial order was set to max. allowed value: {self.filter_order}')
                    print('')
            # Compute data based on new parameter value
            self.compute_data()
            # Update y-lim
            self.y_lim = np.max(self.data.max())
            ax[1].set_ylim([y_lim_min, self.y_lim])
            l.set_ydata(self.data[self.f_rois[self.ind]])
            plt.draw()

        def update_filter_order_slider(self, val):
            self.filter_order = val
            if self.filter_win > 0:  # Filter is ON
                # Check if order > window size - 1
                if self.filter_order >= self.filter_win - 1:
                    print('')
                    print('---------- WARNING ----------')
                    print(f'Window Size to small for polynomial order of: {self.filter_order}')
                    self.filter_win = self.filter_order + 2
                    print(f'Window Size was increased to min. allowed value: {self.filter_win}')
                    print('')
            # Compute data based on new parameter value
            self.compute_data()
            # Update y-lim
            self.y_lim = np.max(self.data.max())
            ax[1].set_ylim([y_lim_min, self.y_lim])
            l.set_ydata(self.data[self.f_rois[self.ind]])
            plt.draw()

        def compute_data(self):
            if self.filter_order <= 0:
                self.filter_order = 1
            if self.filter_win <= 0:
                self.filter_win = 0  # this means no filter
            if self.df_per == 0:  # this means use raw data
                ax[1].set_ylabel('Raw Values')
                self.data = filter_raw_data(f_raw_norm, win=self.filter_win, o=self.filter_order)
            else:
                ax[1].set_ylabel('dF/F')
                self.data, self.f_rois, self.fbs = compute_df_over_f(
                    f_values=filter_raw_data(f_raw, win=self.filter_win, o=self.filter_order),
                    window_size=self.df_win,
                    per=self.df_per,
                    fast=True
                    )

        def good_cell(self, event):
            if self.good_cells_list[self.ind] == 0:
                self.good_cells_list[self.ind] = 1
                self.flip_button(status=self.good_cells_list[self.ind])
            else:
                self.good_cells_list[self.ind] = 0
                self.flip_button(status=self.good_cells_list[self.ind])

        def flip_button(self, status):
            if status == 1:
                # Switch Button
                b_good_cell.label.set_text("GOOD CELL")
                rr = self.f_rois[self.ind]
                ax[0].set_title(f'GOOD ROI: {rr} ({len(self.f_rois)})', color='green')

                # b_good_cell.color = 'green'
                # b_good_cell.hovercolor = 'red'
            else:
                # Switch Button
                b_good_cell.label.set_text("BAD CELL")
                rr = self.f_rois[self.ind]
                ax[0].set_title(f'ROI: {rr} ({len(self.f_rois)})', color='black')

                # b_good_cell.color = 'red'
                # b_good_cell.hovercolor = 'red'

        def export_good_cells(self, event):
            save_dir = select_dir()
            f_list = os.listdir(save_dir)
            file_name = f'{rec_name}_good_cells.csv'
            # check if there already is a list of good cells
            dummy = [s for s in f_list if file_name in s]
            if dummy:
                answer = messagebox.askyesno(title='Overwrite', message='Overwrite ?')
            else:
                answer = True
            if answer:
                good_cells_df = pd.DataFrame()
                good_cells_df['roi'] = np.arange(1, len(f_rois) + 1, 1)
                good_cells_df['quality'] = self.good_cells_list
                good_cells_df.to_csv(f'{save_dir}/{file_name}', index=False)
                messagebox.showinfo(title='Good Cells', message='Successfully exported good cells')
                print('GOOD CELLS EXPORTED!')
            else:
                print('GOOD CELLS NOT EXPORTED!')

    callback = Index()

    # [left, bottom, width, height]
    # Filter Order Slider
    filter_order_slider_ax = plt.axes([0.2, 0.8, 0.1, 0.1])
    filter_order_slider = Slider(
        filter_order_slider_ax,
        'order',
        valmin=1,
        valmax=8,
        valinit=3,
        valstep=1,
        orientation="horizontal",
        color='black'
    )
    filter_order_slider.on_changed(callback.update_filter_order_slider)

    # Filter Window Slider
    # only odd numbers!
    filter_win_slider_ax = plt.axes([0.12, 0.4, 0.02, 0.4])
    filter_win_slider = Slider(
        filter_win_slider_ax,
        'filter',
        valmin=0,
        valmax=51,
        valinit=0,
        valstep=3,
        orientation="vertical",
        color='black'
    )
    filter_win_slider.on_changed(callback.update_filter_win_slider)

    # FBS Percentile Slider
    fbs_per_slider_ax = plt.axes([0.08, 0.4, 0.02, 0.4])
    fbs_per_slider = Slider(
        fbs_per_slider_ax,
        'per',
        valmin=0,
        valmax=100,
        valinit=0,
        valstep=1,
        orientation="vertical",
        color='black'
    )
    fbs_per_slider.on_changed(callback.update_fbs_per_slider)

    # FBS Window Size Slider
    # Create a plt.axes object to hold the slider
    fbs_slider_ax = plt.axes([0.05, 0.4, 0.02, 0.4])
    # Add a slider to the plt.axes object
    fbs_slider = Slider(
        fbs_slider_ax,
        'b win',
        valmin=0,
        valmax=int(np.floor(sig_t[-1])),
        valinit=0,
        valstep=4,
        orientation="vertical",
        color='black'
    )
    fbs_slider.on_changed(callback.update_fbs_win_slider)

    # Add Buttons to Figure
    button_width = 0.05
    button_height = 0.03
    button_font_size = 6

    ax_export_cells = fig.add_axes([0.2, 0.05, button_width, button_height])
    b_export_cells = Button(ax_export_cells, 'EXPORT GOOD CELLS')
    b_export_cells.label.set_fontsize(button_font_size)
    b_export_cells.on_clicked(callback.export_good_cells)

    ax_good_cell = fig.add_axes([0.3, 0.05, button_width, button_height])
    b_good_cell = Button(ax_good_cell, 'BAD CELL')
    b_good_cell.label.set_fontsize(button_font_size)
    # b_good_cell.color('red')
    # b_good_cell.hovercolor = 'green'
    b_good_cell.on_clicked(callback.good_cell)

    ax_next = fig.add_axes([0.5, 0.05, button_width, button_height])
    b_next = Button(ax_next, 'Next')
    b_next.label.set_fontsize(button_font_size)
    b_next.on_clicked(callback.next)

    ax_prev = fig.add_axes([0.4, 0.05, button_width, button_height])
    b_prev = Button(ax_prev, 'Previous')
    b_prev.label.set_fontsize(button_font_size)
    b_prev.on_clicked(callback.prev)
    plt.show()


def plot_mean_response(sig_m, sig_std, sig_t, st, st_t, show):
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(sig_t, sig_m, 'k', alpha=0.5, lw=0.75)
    ax[0].plot(sig_t, sig_m+sig_std, 'k--', alpha=0.5, lw=0.75)
    ax[0].plot(sig_t, sig_m-sig_std, 'k--', alpha=0.5, lw=0.75)

    filtered_m = savitzky_golay(y=sig_m.to_numpy(), window_size=11, order=1)
    filtered_std = savitzky_golay(y=sig_std.to_numpy(), window_size=11, order=1)
    ax[0].plot(sig_t, filtered_m, 'k')
    ax[0].plot(sig_t, filtered_m+filtered_std, 'k--')
    ax[0].plot(sig_t, filtered_m-filtered_std, 'k--')

    ax[1].plot(st_t, st, 'k')
    ax[1].set_xlabel('Time [s]')
    if show:
        plt.show()


def convert_samples_to_time(sig, fr):
    t_out = np.linspace(0, len(sig) / fr, len(sig))
    return t_out


def down_sample(sig, d_rate=10):
    sig_down = sig[d_rate::]
    return sig_down


def debugging():
    embed()
    exit()


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


def compute_z_scores(data):
    return (data - data.mean()) / data.std()


def filter_raw_data(sig, win=11, o=1):
    if win == 0:  # do not filter
        return sig
    # Check if win is an odd number
    if win % 2 == 0:
        win += 1
    a = []
    if len(sig.shape) == 1:
        max_range = sig.shape[0]
    else:
        max_range = sig.shape[1]
    for k in range(max_range):
        a.append(savitzky_golay(y=sig.iloc[:, k].to_numpy(), window_size=win, order=o))
    out = pd.DataFrame(np.transpose(a), columns=sig.keys())
    return out


def find_stimulus_time(volt_threshold, f_stimulation, mode, th_interval_factor):
    # Find stimulus time points
    if mode == 'below':
        threshold_crossings = np.diff(f_stimulation['Volt'] < volt_threshold, prepend=False)
    else:
        mode = 'above'
        threshold_crossings = np.diff(f_stimulation['Volt'] > volt_threshold, prepend=False)

    # Get Upward Crossings
    f_upward = np.argwhere(threshold_crossings)[::2, 0]  # Upward crossings

    # Get Downward Crossings
    f_downward = np.argwhere(threshold_crossings)[1::2, 0]  # Downward crossings

    if th_interval_factor > 0:
        # Remove to small intervals
        f_upward = thresholding_small_intervals(f_upward, f_stimulation, th_factor=th_interval_factor)
        # Remove to small intervals
        f_downward = thresholding_small_intervals(f_downward, f_stimulation, th_factor=th_interval_factor)
    return f_downward, f_upward


def thresholding_small_intervals(data, f_stimulation, th_factor):
    # Threshold for too small intervals
    threshold_intervals = int(np.mean(np.diff(data)) / th_factor)
    print(f'Threshold for Small Intervals: {threshold_intervals}')
    idx = np.diff(data) > threshold_intervals
    idx = np.insert(idx, 0, True)
    stimulus_index_onset_points = data[idx]
    stimulus_onset_times = f_stimulation['Time'].iloc[stimulus_index_onset_points]
    return stimulus_onset_times, stimulus_index_onset_points


def msg_box(f_header, f_msg, sep, r=30):
    print(f'{sep * r} {f_header} {sep * r}')
    print(f'{sep * 2} {f_msg}')
    print(f'{sep * r}{sep}{sep * len(f_header)}{sep}{sep * r}')


def manual_data_selection(data, sort_samples):
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
            msg_box('INFO', f'FOLLOWING POSITIONS SELECTED: {f_samples}', '-')
        else:
            msg_box('WARNING', 'NUMBER OF ENTRIES IS ODD, THEREFORE COULD NOT SORT INTO PAIRS', '-')
    else:
        print('FOLLOWING POSITIONS SELECTED:')
        print(f_samples)
    return f_samples


def get_files_by_suffix(b_dir, tags):
    file_list = os.listdir(b_dir)
    tag_l = dict.fromkeys(tags)
    for tt in tags:
        tag_l[tt] = sorted([s for s in file_list if f'{tt}' in s])

    return tag_l


def select_dir():
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    rec_path = askdirectory()
    return rec_path


def select_file(file_types):
    # file_types example: [('Recording Files', 'raw.txt')]
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    file_name = askopenfilename(filetypes=file_types)
    return file_name


def find_pos_of_char_in_string(str, char):
    idx = [i for i, s in enumerate(str) if char in s]
    return idx


def dict_depth(dic):
    str_dic = str(dic)
    counter = 0
    for i in str_dic:
        if i == "{":
            counter += 1
    return counter


def draw_rois_zipped(img, rois_dic, active_roi, r_color, alp, thickness=1):
    """
        It will draw the ROI as transparent window with border
        Args:
            img   : Image which you want to draw the ROI
            rois_dic: dictionary of rois from zipped rois from ImageJ
            active_roi: Active roi that will be highlighted
            r_color : in RGB (R, G, B), Black = (0, 0, 0)
            alp   : Alpha or transparency value between 0.0 and 1.0
            thickness: thickness in px

        Returns:
            Retuns the processed image
            out = drawROI()
    """

    overlay = img.copy()
    image_out = img.copy()

    if dict_depth(rois_dic) == 1:
        print('SINGLE ROI')
        # convert top, left to center coordinates
        x_coordinate_centered = rois_dic['left'] + rois_dic['width'] // 2
        y_coordinate_centered = rois_dic['top'] + rois_dic['height'] // 2
        center_coordinates = (x_coordinate_centered, y_coordinate_centered)

        # Convert total to half-width for plotting ellipse
        axes_length = (rois_dic['width'] // 2, rois_dic['height'] // 2)

        # Set rest of the parameters for ellipse
        angle = 0
        start_ngle = 0
        end_angle = 360

        # Using cv2.ellipse() method
        # Draw a ellipse with line borders
        cv2.ellipse(overlay, center_coordinates, axes_length,
                    angle, start_ngle, end_angle, r_color, thickness)

        # add ellipse overlay to image
        cv2.addWeighted(overlay, alp, image_out, 1 - alp, 0, image_out)
    else:
        for key in rois_dic:
            if key != active_roi['name']:  # ignore active roi
                # convert top, left to center coordinates
                x_coordinate_centered = rois_dic[key]['left'] + rois_dic[key]['width']//2
                y_coordinate_centered = rois_dic[key]['top'] + rois_dic[key]['height']//2
                center_coordinates = (x_coordinate_centered, y_coordinate_centered)

                # Convert total to half-width for plotting ellipse
                axes_length = (rois_dic[key]['width'] // 2, rois_dic[key]['height'] // 2)

                # Set rest of the parameters for ellipse
                angle = 0
                start_angle = 0
                end_angle = 360

                # Using cv2.ellipse() method
                # Draw a ellipse with line borders
                cv2.ellipse(overlay, center_coordinates, axes_length,
                            angle, start_angle, end_angle, r_color, thickness)

    # ACTIVE ROI
    # convert top, left to center coordinates
    x_coordinate_centered = active_roi['left'] + active_roi['width'] // 2
    y_coordinate_centered = active_roi['top'] + active_roi['height'] // 2
    center_coordinates = (x_coordinate_centered, y_coordinate_centered)

    # Convert total to half-width for plotting ellipse
    axes_length = (active_roi['width'] // 2, active_roi['height'] // 2)

    # Set rest of the parameters for ellipse
    angle = 0
    start_angle = 0
    end_angle = 360

    # Using cv2.ellipse() method
    # Draw a ellipse with line borders
    r_color = (255, 0, 0)
    cv2.ellipse(overlay, center_coordinates, axes_length,
                angle, start_angle, end_angle, r_color, thickness)

    # add ellipse overlays to image
    cv2.addWeighted(overlay, alp, image_out, 1 - alp, 0, image_out)

    return image_out


def add_roi_labels(f_ax, rois_dic, a_roi_nr, color=(1, 1, 1), size=5):
    for key in rois_dic:
        # center coordinates
        x_pos = rois_dic[key]['left'] + rois_dic[key]['width'] // 2
        y_pos = rois_dic[key]['top'] + rois_dic[key]['height']//2
        # find roi nr
        idx = find_pos_of_char_in_string(rois_dic[key]['name'], '-')[0]
        roi_nr = int(rois_dic[key]['name'][0:idx])
        if roi_nr == a_roi_nr:
            use_color = (1, 0, 0)
            font_size = size + 3
        else:
            use_color = color
            font_size = size

        f_ax.text(
            x_pos, y_pos, roi_nr, fontsize=font_size, color=use_color,
            horizontalalignment='center', verticalalignment='center'
        )


def rename_nw_stimulus_files(b_dir, do_copy=True, do_remove=False):
    tag = 'tactile.txt'
    f_list = os.listdir(b_dir)
    pos = [0, 2]
    f_names = [s for s in f_list if tag in s]
    n = []
    for st in f_names:
        seps = find_pos_of_char_in_string(st, '_')
        if len(seps) == 5:
            new_name = f'{st[0:seps[pos[0]]]}_{st[seps[pos[1]] + 3:seps[pos[1]] + 2 + 4]}_{tag}'
            n.append(new_name)
            src = f'{b_dir}/{st}'
            dst = f'{b_dir}/{new_name}'
            if do_copy:
                # rename: shutil.copyfile(source, destination)
                shutil.copyfile(src, dst)
            if do_remove:
                os.remove(src)
        elif len(seps) == 4:
            new_name = f'{st[0:seps[pos[0]]]}_{st[seps[pos[1]] + 3:seps[pos[1]] + 2 + 2]}_1_{tag}'
            n.append(new_name)
            src = f'{b_dir}/{st}'
            dst = f'{b_dir}/{new_name}'
            # rename: shutil.copyfile(source, destination)
            if do_copy:
                shutil.copyfile(src, dst)
            if do_remove:
                os.remove(src)
        else:
            print('WRONG FILE NAME')
    return n, f_names


def rename_nw_result_files(b_dir, do_copy=True, do_remove=False):
    tag = 'results'
    f_list = os.listdir(b_dir)
    f_names = [s for s in f_list if tag in s]
    n = []
    for st in f_names:
        seps = find_pos_of_char_in_string(st, '_')
        if len(seps) == 4:
            new_name = f'{st[seps[-3]+1:-4]}_raw.txt'
            n.append(new_name)
            src = f'{b_dir}/{st}'
            dst = f'{b_dir}/{new_name}'
            # rename: shutil.copyfile(source, destination)
            if do_copy:
                shutil.copyfile(src, dst)
            if do_remove:
                os.remove(src)
        elif len(seps) == 3:
            new_name = f'{st[seps[-2]+1:-4]}_1_raw.txt'
            n.append(new_name)
            src = f'{b_dir}/{st}'
            dst = f'{b_dir}/{new_name}'
            # rename: shutil.copyfile(source, destination)
            if do_copy:
                shutil.copyfile(src, dst)
            if do_remove:
                os.remove(src)
        else:
            print('WRONG FILE NAME')
    return n, f_names


def rename_nw_ref_files(b_dir, do_copy=True, do_remove=False):
    b_dir = f'{b_dir}/refs'
    tag = 'ROI.tif'
    f_list = os.listdir(b_dir)
    pos = [0, 2]
    f_names = [s for s in f_list if tag in s]
    n = []
    for st_0 in f_names:
        if st_0.startswith('STD'):
            # Remove "STD_"
            st = st_0[4:]
        else:
            st = st_0
        seps = find_pos_of_char_in_string(st, '_')
        if len(seps) == 6:
            new_name = f'{st[0:seps[pos[0]]]}_{st[seps[pos[1]] + 3:seps[pos[1]] + 2 + 4]}_{tag}'
            n.append(new_name)
            src = f'{b_dir}/{st_0}'
            dst = f'{b_dir}/{new_name}'
            # rename: shutil.copyfile(source, destination)
            if do_copy:
                shutil.copyfile(src, dst)
            if do_remove:
                os.remove(src)
        elif len(seps) == 5:
            new_name = f'{st[0:seps[pos[0]]]}_{st[seps[pos[1]] + 3:seps[pos[1]] + 2 + 2]}_1_{tag}'
            n.append(new_name)
            src = f'{b_dir}/{st_0}'
            dst = f'{b_dir}/{new_name}'
            # rename: shutil.copyfile(source, destination)
            if do_copy:
                shutil.copyfile(src, dst)
            if do_remove:
                os.remove(src)
        else:
            print('WRONG FILE NAME')
    return n, f_names


def create_binary_trace(sig, cirf, start, end, fr, ca_delay, pad_before, pad_after, low=0, high=1):
    # start and end are onset and offset time points
    # fr is sampling rate of the ca imaging recording
    # Convert onset and offset time into samples (ca recording)
    onset_samples = np.array(start * fr).astype('int')
    offset_samples = np.array(end * fr).astype('int')

    # check if all cutouts have same duration
    template = offset_samples[0] - onset_samples[0]
    for i in range(len(onset_samples)):
        dur = offset_samples[i] - onset_samples[i]
        check = template == dur
        if not check:
            print('Not all Cutouts have same duration ... Will correct for that')
            diff = template - dur
            offset_samples[i] = offset_samples[i] + diff
            print(f'Diff: {diff} samples')

    # max_dur_samples = np.array(max_dur * fr).astype('int')/
    max_dur_samples = int(len(sig))
    # Trace with only "low" values (e.g. zeros)
    binary_trace = np.zeros(max_dur_samples) + low

    # Delay stimulus times to compensate for slow GCaMP dynamics and neuronal delay
    onset_samples = onset_samples + int(ca_delay * fr)
    offset_samples = offset_samples + int(ca_delay * fr)

    # Insert "high" values (e.g. ones) where there is a stimulus (range between onset and offset)
    for kk in zip(onset_samples, offset_samples):
        idx1 = kk[0]
        idx2 = kk[1]
        if idx2 <= idx1:  # if the duration is below the ca imaging time resolution (e.g. 2 Hz => 500 ms)
            idx2 += 1
        binary_trace[idx1:idx2] = high

    # Convolve Binary with cirf
    reg = np.convolve(binary_trace, cirf, 'full')

    # onset_samples = np.array(start * fr).astype('int')
    # offset_samples = np.array(end * fr).astype('int')

    # Add padding for cutouts
    onset_samples = onset_samples - int(pad_before * fr)
    offset_samples = offset_samples + int(pad_after * fr)



    # if offset_samples[-1] >= max_dur_samples:
    #   offset_samples[-1] = max_dur_samples
    #   print('Last Cutout exceeded stimulus duration and cutout was corrected for that!')

    # Cut out windows
    sig_cut_outs = []
    reg_cut_outs = []
    for kk in zip(onset_samples, offset_samples):
        # cut out ca responses
        sig_cut_outs.append(sig.iloc[kk[0]:kk[1]])
        # cut out regressor
        reg_cut_outs.append(reg[kk[0]:kk[1]])
    return binary_trace, reg, sig_cut_outs, reg_cut_outs


def create_cif(fr, tau):
    t_max = tau * 5  # in sec
    t_cif = np.arange(0, t_max*fr, 1)
    tau_samples = tau * fr
    cif = np.exp(-(t_cif/tau_samples))
    return cif


def apply_linear_model(xx, yy, norm_reg=True):
    # Normalize data to [0, 1]
    if norm_reg:
        f_y = yy / np.max(yy)
    else:
        f_y = yy

    # Check dimensions of reg
    if len(xx.shape) == 1:
        reg_xx = xx.reshape(-1, 1)
    elif len(xx.shape) == 2:
        reg_xx = xx
    else:
        print('ERROR: Wrong x input')
        return False

    # Linear Regression
    l_model = LinearRegression().fit(reg_xx, f_y)

    # Slope (y = a * x + c)
    a = l_model.coef_[0]
    # R**2 of model
    f_r_squared = l_model.score(reg_xx, f_y)
    # Score
    f_score = a * f_r_squared
    return f_score, f_r_squared, a
