import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import analysis_util_functions as uf
from IPython import embed
from read_roi import read_roi_zip
import os
import sys

uf.msg_box(
    'INFO',
    'You can give extra arguments to show only selected cells:\n -good : good cells (LM Score) '
    , sep='+'
)

show_selection = False
if len(sys.argv) > 1:
    command = sys.argv[1]

    if command == '-good':
        show_selection = True

# Select Directory and get all files
file_name = uf.select_file([('Recording Files', 'raw.txt')])
data_file_name = os.path.split(file_name)[1]
rec_dir = os.path.split(file_name)[0]
rec_name = os.path.split(rec_dir)[1]
# rec_name = os.path.split(file_name)[1][0:uf.find_pos_of_char_in_string(os.path.split(file_name)[1], '_')[-1]]
uf.msg_box(rec_name, f'SELECTED RECORDING: {rec_name}', '+')

# Look for good cell list
found_good_cells = False
for kk in os.listdir(rec_dir):
    if 'good_cells.csv' in kk:
        found_good_cells = True

if found_good_cells:
    good_cells_list = pd.read_csv(f'{rec_dir}/{rec_name}_good_cells.csv')
else:
    good_cells_list = []

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

# Import Linear Regression Scoring
good_cells_by_score_csv = pd.read_csv(f'{rec_dir}/{rec_name}_lm_good_score_rois.csv')
final_mean_score = pd.read_csv(f'{rec_dir}/{rec_name}_lm_mean_scores.csv')
all_cells = np.load(f'{rec_dir}/{rec_name}_lm_results.npy', allow_pickle=True).item()
score_th = 0.15

# Select Good Data
if show_selection:
    if good_cells_by_score_csv['roi'].shape[0] == 0:
        print('')
        uf.msg_box('WARNING', 'There are no cells with high enough LM Scores!', '-')
        pd.DataFrame().to_csv(f'{rec_dir}/{rec_name}_NO_GOOD_CELLS.csv')
        exit()
    data = f_raw[good_cells_by_score_csv['roi']]
else:
    data = f_raw

# Start Data Viewer
uf.data_viewer(
    rec_name=rec_name,
    f_raw=data,
    sig_t=roi_time_axis,
    ref_im=img_ref,
    st_rec=stimulus,
    protocol=protocol,
    rois_dic=rois_in_ref,
    good_cells_list=good_cells_list,
    good_scores=good_cells_by_score_csv,
    scores=final_mean_score,
    score_th=score_th
)
