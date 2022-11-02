import pandas as pd
import numpy as np
import analysis_util_functions as uf
import os
from IPython import embed
import csv


def import_f_raw(file_dir):
    # Check for header and delimiter
    with open(file_dir) as csv_file:
        dialect = csv.Sniffer().sniff(csv_file.read(32))
        delimiter = dialect.delimiter

    # Load data and assume the first row is the header
    data = pd.read_csv(file_dir, decimal='.', sep=delimiter, header=0, index_col=0).reset_index(drop=True)
    # Chek for correct header
    try:
        a = float(data.keys()[0])
        data = pd.read_csv(file_dir, decimal='.', sep=delimiter, header=None)
    except ValueError:
        a = 0

    header_labels = []
    for kk in range(data.shape[1]):
        header_labels.append(f'roi_{kk + 1}')
    data.columns = header_labels
    return data


def extend_stimulus(f_stimulus, padding=30, fr=1000):
    # Padding in secs
    # Add Before Recording
    insertion = pd.DataFrame()
    insertion['Time'] = np.linspace(0, padding, padding * fr)
    insertion['Volt'] = np.zeros(len(insertion['Time']))
    f_stimulus = pd.concat([insertion, f_stimulus.iloc[1:]], ignore_index=True)

    # Add after Recording
    insertion = pd.DataFrame()
    insertion['Time'] = np.linspace(f_stimulus['Time'].max() + 1, f_stimulus['Time'].max() + padding, padding * fr)
    insertion['Volt'] = np.zeros(len(insertion['Time']))
    f_stimulus = pd.concat([f_stimulus, insertion])

    return f_stimulus


def extend_f_raw(data, padding=30, fr=2.0345147125756804):
    # padding in secs
    # f_raw must be a pandas data frame with rois as header

    # Add Before Recording
    # Convert to samples
    time_added_rec_samples = int(padding * fr)
    rois_count = len(data.keys())
    added_values = data.iloc[:3, :].mean()
    f_insertion = pd.DataFrame(np.zeros((time_added_rec_samples, rois_count)), columns=data.keys())
    data = pd.concat([f_insertion + added_values, data], ignore_index=True)

    # Add After Recording
    data = pd.concat([data, f_insertion + added_values], ignore_index=True)

    return data


# SELECT DATA ----------------------------------------------------------------------------------------------------------
uf.msg_box('INFO', 'STARTING PREPARING RECORDING', sep='+')
rec_dir = uf.select_dir()
rec_name = os.path.split(rec_dir)[1]
# Import Stimulus File (tab separated and no headers as it comes from MOM Sutter)
stimulation_original = pd.read_csv(f'{rec_dir}/{rec_name}_stimulation_sutter.txt', decimal='.', sep='\t', header=None)
stimulation_original.columns = ['Time', 'Volt']

# Import Raw Data (delimiter and headers are automatically estimated)
f_raw_original = import_f_raw(f'{rec_dir}/{rec_name}_raw_imagej.txt')

# Store to HDD as a Backup
stimulation_original.to_csv(f'{rec_dir}/{rec_name}_stimulation_original.txt', index=None)
f_raw_original.to_csv(f'{rec_dir}/{rec_name}_raw_original.txt', index=None)

# Extend Recording
# Check if Recording has already been extended
file_list = [s for s in os.listdir(rec_dir) if 'RECORDING_WAS_EXTENDED' in s]
if len(file_list) == 0:
    # Now Extend the Recording just to be save ...
    print('Extending Recording ...')
    print('')
    f_raw = extend_f_raw(f_raw_original)
    stimulation_file = extend_stimulus(stimulation_original)

    # Store extended to HDD
    f_raw.to_csv(f'{rec_dir}/{rec_name}_raw.txt', index=None)
    stimulation_file.to_csv(f'{rec_dir}/{rec_name}_stimulation.txt', index=None)
    pd.DataFrame().to_csv(f'{rec_dir}/{rec_name}_RECORDING_WAS_EXTENDED.txt', decimal='.', sep='\t',
                          header=None)

    uf.msg_box('INFO', 'RECORDING HAS BEEN EXTENDED AND STORED TO HDD (+BACKUPS)', sep='+')
    print('')
else:
    uf.msg_box('INFO', 'RECORDING IS ALREADY EXTENDED', sep='+')
    print('')

uf.msg_box('INFO', 'STIMULUS AND RAW DATA HAVE BEEN PREPARED FOR FURTHER ANALYSIS', sep='+')
