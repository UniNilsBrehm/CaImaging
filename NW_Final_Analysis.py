import analysis_util_functions as uf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

file_dir = uf.select_file([('CSV Files', '.csv')])
df = pd.read_csv(file_dir, index_col=0)

embed()
exit()

# Collect responses to step 100
idx_step_100 = (df['stimulus_type'] == 'Step') \
               & (df['stimulus_parameter'] == 100) \
               & (df['anatomy'] == 'tg') \
               & (df['mean_score'] > 0)

data = df[idx_step_100].copy()
cell_names = data['id'].unique()
a = []
for c in cell_names:
    d = data[data['id'] == c].copy()
    for tr in d['trial'].unique():
        idx = d['trial'] == tr
        size = len(d.loc[(idx, 'df')])
        f = d.loc[(idx, 'df')].to_numpy()
        z = (f - np.mean(f))/np.std(f)
        a.append(z)

plt.figure()
plt.title(f'n={cell_names.shape[0]}')
m = np.mean(a, axis=0)
sem = np.std(a, axis=0) / np.sqrt(len(m))
plt.plot(m, 'k')
plt.plot(m-sem, 'r')
plt.plot(m+sem, 'r')
plt.show()
