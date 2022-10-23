import analysis_util_functions as uf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython import embed

file_dir = uf.select_file([('CSV Files', '.csv')])
df = pd.read_csv(file_dir, index_col=0)

embed()
exit()
time_before = 5
time_after = 20

idx = (df['stimulus_onset_type'] == 'Step') \
      & (df['stimulus_parameter'] == 100) \
      & (df['anatomy'] == 'tg') \
      & (df['mean_score'] > 0.2)

idx_ramp = np.where(idx == True)[0]
cell_names = df.loc[idx]['id'].unique()
z_scores = [[]] * len(idx_ramp)
fr = df.iloc[idx_ramp[0]]['fr']
for k, s in enumerate(idx_ramp):
    before = int(time_before * fr)
    after = int(time_after * fr)
    z = df.iloc[s-before:s+after]['z-score']
    z_scores[k] = z

plt.figure()
plt.title(f'n={cell_names.shape[0]}')
m = np.mean(z_scores, axis=0)
sem = np.std(z_scores, axis=0) / np.sqrt(len(m))
t_axis = uf.convert_samples_to_time(sig=m, fr=fr)
plt.plot(t_axis, m, 'k')
plt.plot(t_axis, m-sem, 'r')
plt.plot(t_axis, m+sem, 'r')
plt.show()

exit()

# # Compute z scores
# cells_unique = df['id'].unique()
# for cell in cells_unique:
#     # Get Raw Values
#     r = df.loc[(df['id'] == cell, 'raw')]
#     # Filter Raw Values
#     r_fil = uf.savitzky_golay(y=r.to_numpy(), window_size=5, order=2)
#     r_fil = pd.DataFrame(r_fil)
#     # Compute delta f over f
#     f_df, f_rois, fbs = uf.compute_df_over_f(f_values=r_fil, window_size=100, per=5, fast=True)
#     # fbs = np.round(np.percentile(r_fil, 5, axis=0), 3)
#     # f_df = (r_fil - fbs) / fbs
#     # Compute z score
#     z = (f_df-np.mean(f_df, axis=0)) / np.std(f_df, axis=0)
#     df.loc[(df['id'] == cell, 'z_score')] = z.to_numpy()

# save_dir = os.path.split(file_dir)[0]
# df.to_csv(f'{save_dir}/data_frame2.csv')
