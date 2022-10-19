import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import analysis_util_functions as uf
from IPython import embed
import multipagetiff as mtif
from read_roi import read_roi_zip
import cv2
import os

""" This Script will perform a Linear Regression Model between Cell Responses (Ca-Imaging) and Stimulus Trace to get
    a score for each cell in the recording 
"""
# SETTINGS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
score_th = 0.15
cirf_tau = 10  # in secs
filter_window = 12  # in secs
filter_order = 2
fbs_percentile = 5  # in %
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Select Directory and get all files
file_name = uf.select_file([('Recording Files', 'raw.txt')])
data_file_name = os.path.split(file_name)[1]
rec_dir = os.path.split(file_name)[0]
rec_name = os.path.split(rec_dir)[1]
# rec_name = os.path.split(file_name)[1][0:uf.find_pos_of_char_in_string(os.path.split(file_name)[1], '_')[-1]]
uf.msg_box(rec_name, f'SELECTED RECORDING: {rec_name}', '+')

# Import stimulation trace
stimulus = uf.import_txt_stimulation_file(f'{rec_dir}', f'{rec_name}_stimulation', float_dec='.')

# Import Protocol
protocol = pd.read_csv(f'{rec_dir}/{rec_name}_protocol.csv')

# Import Reference Image
img_ref = plt.imread(f'{rec_dir}/refs/{rec_name}_ROI.tif.jpg', format='jpg')

# Import ROIS from Imagej
rois_in_ref = read_roi_zip(f'{rec_dir}/refs/{rec_name}_ROI.tif_RoiSet.zip')

# Import raw fluorescence traces (rois)
# It is important that the header is equal to the correct ROI number
# All ROIS of the recording
header_labels = []
for k, v in enumerate(rois_in_ref):
    header_labels.append(f'roi_{k+1}')
f_raw = pd.read_csv(f'{rec_dir}/{data_file_name}', decimal='.', sep='\t', index_col=0, header=None)
f_raw.columns = header_labels

# Estimate frame rate
fr_rec = uf.estimate_sampling_rate(data=f_raw, f_stimulation=stimulus, print_msg=False)

# Correct for too short recordings
pad_after = 20  # in secs
diff = stimulus['Time'].max() - protocol['Offset_Time'].max()
if diff <= pad_after:
    print('Stimulus Recording too short ... Will correct for that ...')
    t = np.linspace(stimulus['Time'].max()+1/1000, stimulus['Time'].max()+pad_after*2, 1000)
    v = np.zeros(len(t))
    additional_stimulus = pd.DataFrame()
    additional_stimulus['Time'] = t
    additional_stimulus['Volt'] = v

    tt = int(pad_after * 2 * fr_rec)
    w = np.zeros((tt, len(f_raw.keys()))) + np.percentile(f_raw, 5)
    additional_recording = pd.DataFrame(w, columns=f_raw.keys())

    # add this to original recording
    stimulus = pd.concat([stimulus, additional_stimulus])
    f_raw = pd.concat([f_raw, additional_recording])

# Compute time axis for rois
roi_time_axis = uf.convert_samples_to_time(sig=f_raw, fr=fr_rec)

# Get step and ramp stimuli
step_parameters = protocol[protocol['Stimulus'] == 'Step']['Duration'].unique()
ramp_parameters = protocol[protocol['Stimulus'] == 'Ramp']['Duration'].unique()

stimulus_parameters = pd.DataFrame()
stimulus_parameters['parameter'] = np.append(step_parameters, ramp_parameters)
stimulus_parameters['type'] = np.append(['Step'] * len(step_parameters), ['Ramp'] * len(ramp_parameters))

# Compute delta f over f
f_raw_filtered = uf.filter_raw_data(sig=f_raw, win=filter_window, o=filter_order)
fbs = np.percentile(f_raw_filtered, fbs_percentile, axis=0)
df_f = (f_raw_filtered-fbs) / fbs

# Compute Calcium Impulse Response Function (CIRF)
cirf = uf.create_cif(fr_rec, tau=cirf_tau)

# Select Stimulus Type
label = []
for k in range(stimulus_parameters.shape[0]):
    label.append(f'{stimulus_parameters.iloc[k, 1]}-{stimulus_parameters.iloc[k, 0]}')

# Create Binary and cut out ROI Ca responses for each stimulus type
stimulus_regression = []
for k in range(stimulus_parameters.shape[0]):
    stim = protocol[(protocol['Stimulus'] == stimulus_parameters.iloc[k, 1]) &
                    (protocol['Duration'] == stimulus_parameters.iloc[k, 0])]
    binary, reg, response_c, reg_c = uf.create_binary_trace(
        sig=df_f, cirf=cirf, start=stim['Onset_Time'], end=stim['Offset_Time'],
        fr=fr_rec, ca_delay=0, pad_before=5, pad_after=20, low=0, high=1
    )
    dummy = {'binary': binary, 'reg': reg, 'response_cutout': response_c, 'reg_cutout': reg_c}
    stimulus_regression.append(dummy)

# Linear Regression Model to Score Responses to Stimulus Types
all_cells = dict.fromkeys(df_f.keys())
for rois in df_f:  # loop through rois (cells)
    lm = []
    lm2 = dict.fromkeys(label)
    for k, stim_type in enumerate(stimulus_regression):  # loop through stimulus types (6)
        score, r_squared, slope, response_trace = [], [], [], []
        for kk, repeat_reg in enumerate(stim_type['reg_cutout']):  # loop through repetitions (5)
            dummy_response = stim_type['response_cutout'][kk][rois]
            # Linear Model for one cell, one stimulus type and one repetition
            sc, rr, aa = uf.apply_linear_model(xx=repeat_reg, yy=dummy_response, norm_reg=True)
            score.append(sc)
            r_squared.append(rr)
            slope.append(aa)
            response_trace.append(dummy_response)
        lm.append({'score': score, 'r_squared': r_squared, 'slope': slope, 'response': response_trace})
        lm2[label[k]] = {'score': score, 'r_squared': r_squared, 'slope': slope, 'response': response_trace}
        # all stimulus types for one cell
    all_cells[rois] = lm2
# Access: roi = 27, stimulus type and repetition nr 2
# a = all_cells['27'][Ramp-400]['score'][2]

# Compute Mean Score Values
mean_score = dict.fromkeys(df_f)
mean_r_squared = dict.fromkeys(df_f)
mean_slope = dict.fromkeys(df_f)
for rois in df_f:
    score_stim_type_means = []
    r_squared_stim_type_means = []
    slope_stim_type_means = []
    response_means = []
    for k, stim_type in enumerate(stimulus_regression):
        score_stim_type_means.append(np.mean(all_cells[rois][label[k]]['score']))
        r_squared_stim_type_means.append(np.mean(all_cells[rois][label[k]]['r_squared']))
        slope_stim_type_means.append(np.mean(all_cells[rois][label[k]]['slope']))
        response_means.append(np.mean(all_cells[rois][label[k]]['response']))

    mean_score[rois] = score_stim_type_means
    mean_r_squared[rois] = r_squared_stim_type_means
    mean_slope[rois] = slope_stim_type_means

# Indices are the different stimulus types (6) showing mean score values (all 5 repetitions)
mean_score = pd.DataFrame(mean_score)
mean_r_squared = pd.DataFrame(mean_r_squared)
mean_slope = pd.DataFrame(mean_slope)
final_mean_score = pd.concat([stimulus_parameters, mean_score], axis=1)

# Sort out cells by LM score
good_cells_by_score = []
for key in final_mean_score.keys()[2:]:
    if any(final_mean_score[key] >= score_th):
        good_cells_by_score.append(key)

# Store Linear Regression Results to HDD
# load it with np.load(dir, allow_pickle).item()
good_cells_by_score_csv = pd.DataFrame(good_cells_by_score, columns=['roi'])
good_cells_by_score_csv.to_csv(f'{rec_dir}/{rec_name}_lm_good_score_rois.csv')
np.save(f'{rec_dir}/{rec_name}_lm_results.npy', all_cells)
final_mean_score.to_csv(f'{rec_dir}/{rec_name}_lm_mean_scores.csv')


# Create Anatomy Labels
# check if there is already a anatomy label
check = [s for s in os.listdir(rec_dir) if 'anatomy' in s]
if not check:
    anatomy = pd.DataFrame(columns=good_cells_by_score_csv['roi'])
    anatomy.to_csv(f'{rec_dir}/{rec_name}_anatomy.csv')
else:
    uf.msg_box('INFO', 'FOUND ANATOMY LABELS', '-')
    anatomy = pd.read_csv(f'{rec_dir}/{rec_name}_anatomy.csv')
    print(anatomy)

uf.msg_box('INFO', 'Linear Regression Scoring finished! All Data was stored to HDD.', sep='+')
