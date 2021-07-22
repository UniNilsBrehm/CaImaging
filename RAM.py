import mmap
import os
import time
import datatable
import dask.dataframe
import numpy as np
import pandas as pd
from IPython import embed


def mmap_io(filename):
    output = []
    with open(filename, mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
            text_bytes = mmap_obj.read()
            text = text_bytes.decode('utf-8')
            text_list = text.splitlines()
            for i, v in enumerate(text_list):  # not using readlines(), as this consumes the memory
                text_list[i] = v.split()
            # output = np.array(text)
    return text_list


def pandas_to_ram(filename):
    numpy_array = pd.read_csv(filename, sep='\t', lineterminator='\n', header=None, quoting=2).to_numpy()
    f = numpy_array.tobytes()
    m = mmap.mmap(-1, len(f))
    m.write(f)
    # with mmap.mmap(f.fileno(), 0) as mm:
    #     out = mm.write(f)

    return m


p = 'C:/Uni Freiburg/NilsWenke_Datenauswertung_2021/Raw_data/test/'
files = os.listdir(p)
t1 = time.perf_counter()
out1 = []
for n in files:
    out1.append(dask.dataframe.read_csv(f'{p}{n}'))
t2 = time.perf_counter()

out2 = []
for n in files:
    out2.append(pd.read_csv(f'{p}{n}', sep='\t', lineterminator='\n', header=None, quoting=2).to_numpy())
t3 = time.perf_counter()

out3 = []
for n in files:
    out3.append(datatable.fread(f'{p}{n}').to_numpy())

t4 = time.perf_counter()

print(f'dask: {t2-t1} secs')
print(f'pandas: {t3-t2} secs')
print(f'R datatable: {t4-t3} secs')

embed()
exit()
