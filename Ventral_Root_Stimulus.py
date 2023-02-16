import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tkinter.filedialog import askdirectory
from tkinter import Tk
from IPython import embed

Tk().withdraw()
base_dir = askdirectory()
stimulus_data_dir = f'{base_dir}/stimulation'

file_list = os.listdir(stimulus_data_dir)
# vr_files = [s for s in file_list if "ephys" in s]
stimulus = dict()
for f_name in file_list:
    stimulus[f_name[6:-4]] = pd.read_csv(f'{stimulus_data_dir}/{f_name}', sep='\t', header=None)
rec_name = file_list[0][:5]
idx = list(stimulus.keys())
embed()
exit()
