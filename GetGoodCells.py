import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import analysis_util_functions as uf
from IPython import embed
import os

rec_dir = uf.select_dir()
rec_names = os.listdir(rec_dir)
file_names = uf.get_files_by_suffix(rec_dir, ['good_cells.csv'])['good_cells.csv']

# # Create Directories for all recordings
# for ff in file_names:
#     rec_name = ff[:10]
#     os.makedirs(f'{rec_dir}/{rec_name}/')
# exit()

all_rec_names = []
good_cells_count = 0
good_cells = {}
f_raw = {}
for name in rec_names:
    temp_dir = f'{rec_dir}/{name}/{name}'
    # Read Good Cell Labels
    csv_file = pd.read_csv(f'{temp_dir}_good_cells.csv')

    # Import Raw Data
    data = pd.read_csv(f'{temp_dir}_raw.txt', decimal='.', sep='\t', index_col=0, header=None)
    idx = csv_file['quality'] == 1
    if idx.sum() > 0:
        good_data = data.iloc[:, idx.to_numpy()]
        good_data.to_csv(f'{temp_dir}_good_cells_raw.txt', decimal='.', sep='\t')
        f_raw[name] = good_data
    else:
        no_good = pd.DataFrame()
        no_good.to_csv(f'{temp_dir}_NO_GOOD_RESPONSES.txt')


