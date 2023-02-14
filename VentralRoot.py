import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from IPython import embed


base_dir = "E:/CaImagingAnalysis/20230208_Nils/rec01/"
file_list = os.listdir(base_dir)
file_list = [s for s in file_list if "ephys" in s]

# Load files
vr_data = []
for f_name in file_list:
    print(f_name)
    vr_data.append(pd.read_csv(f'{base_dir}{f_name}', sep='\t', header=None).iloc[:, [0, 3]])

# Concat all to one data frame
vr_trace = pd.concat(vr_data).iloc[:, 0]
vr_time_stamps = pd.concat(vr_data).iloc[:, 1]
# Load ca imaging recording
f_raw = pd.read_csv(f'{base_dir}recording/raw_values.csv', index_col=0)
fr = 2
f_time_axis = np.linspace(0, len(f_raw) / fr, len(f_raw))
vr_time_axis = np.linspace(0, f_time_axis[-1], len(vr_trace))

roi_name = 'Mean1'
# plt.plot(vr_time_axis, vr_trace/np.max(vr_trace), 'r')
# plt.plot(f_time_axis, f_raw[roi_name]/np.max(f_raw[roi_name]), 'k')
# plt.show()

# Store to HDD (Normalized to 0-1)
vr_trace_export = pd.DataFrame(columns=['Time', 'Volt'])
vr_trace_export['Time'] = vr_time_axis
vr_trace_export['Volt'] = (vr_trace.to_numpy() / np.max(vr_trace.to_numpy())) * 10

# Down-sample ventral root recording
vr_trace_export_ds = vr_trace_export[::10]
vr_trace_export_ds.to_csv(f'{base_dir}recording/ventral_root_trace.csv')

print('DONE')
# roi_name = 'Mean1'
# plt.plot(vr_trace_export_ds['Time'], vr_trace_export_ds['Volt'], 'r')
# plt.plot(f_time_axis, f_raw[roi_name]/np.max(f_raw[roi_name]), 'k')
# plt.show()

