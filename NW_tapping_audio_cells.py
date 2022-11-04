import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import embed
import os

rec_dir = 'E:/CaImagingAnalysis/Paper_Data/NilsWenke/TappingAuditoryCells/180418_2_1'
rec_name = os.path.split(rec_dir)[1]
f_raw = pd.read_csv(f'{rec_dir}/{rec_name}_raw.txt')
protocol = pd.read_csv(f'{rec_dir}/{rec_name}_protocol.csv')
anatomy = pd.read_csv(f'{rec_dir}/{rec_name}_anatomy.csv')

