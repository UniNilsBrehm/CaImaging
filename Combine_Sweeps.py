import numpy as np
import pandas as pd
import os
import csv
from analysis_util_functions import select_dir
from IPython import  embed


def import_stimulation_file(file_dir):
    # Check for header and delimiter
    with open(file_dir) as csv_file:
        some_lines = csv_file.read(512)
        dialect = csv.Sniffer().sniff(some_lines)
        delimiter = dialect.delimiter
    # Load data and assume the first row is the header
    data = pd.read_csv(file_dir, decimal='.', sep=delimiter, header=0)
    # Chek for correct header
    try:
        a = float(data.keys()[0])  # this means no headers in the file
        data = pd.read_csv(file_dir, decimal='.', sep=delimiter, header=None)
        data.columns = ['Time', 'Volt']
        return data
    except ValueError:
        data = data.drop(data.columns[0], axis=1)
        return data


def combine_stimulation_sweeps(f_stimulation):
    # Combine stimulation files  of all sweeps into one data frame
    # INPUTS:
    # f_stimulation: list of all stimulation data frames (txt files)

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

    return f_stimulation


def combine_protocol_sweeps(f_protocols):
    # Combine protocol files of all sweeps into one data frame
    # INPUTS:
    # f_protocols: list of all protocol data frames (cvs files)
    # Combine all stimulus protocol logs:
    sweep_names = []
    for ii in range(len(f_protocols)):
        sweep_names.append(f'sweep{ii + 1}')
    f_protocols = pd.concat(f_protocols, keys=sweep_names)
    return f_protocols


if __name__ == '__main__':
    # Import all files
    rec_dir = select_dir()
    file_list = os.listdir(rec_dir)
    stimulation_list = [s for s in file_list if '.txt' in s]
    stimulation_sweeps = []
    for file in stimulation_list:
        stimulation_sweeps.append(import_stimulation_file(f'{rec_dir}/{file}'))
    stimulation = combine_stimulation_sweeps(stimulation_sweeps)
    # Store combined stimulation file
    stimulation.to_csv(f'{rec_dir}/combined_stimulation.txt')
    print('Stimulation Sweeps were combined successfully!')
