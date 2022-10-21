import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import analysis_util_functions as uf
from IPython import embed
from read_roi import read_roi_zip
import os
import sys
import time

base_dir = uf.select_dir()
# Ignore all files and only take directories
rec_list = [s for s in os.listdir(base_dir) if '.' not in s]
uf.msg_box('INFO', 'COLLECTING DATA ... PLEASE WAIT ...', '+')
df = pd.DataFrame()
for rec_name in rec_list:
    df_dummy = pd.read_csv(f'{base_dir}/{rec_name}/data_frame.csv', index_col=0)
    df = pd.concat([df, df_dummy])

uf.msg_box('INFO', f'Data Frame Size: {df.shape[0]:,} rows and {df.shape[1]:,} columns', '+')
# Store Data Frame as csv file to HDD
df.to_csv(f'{base_dir}/data_frame.csv')
uf.msg_box('INFO', f'Stored Data Frame to: {base_dir}/', '+')
