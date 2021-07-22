import matplotlib.pyplot as plt
import numpy as np
import os
import datatable
from scipy import stats
from sklearn.linear_model import LinearRegression
from IPython import embed
import time as stop_watch
import pandas as pd
from multiprocessing import Pool


def define_stimuli(Stimulus, threshold=0.012, deadtime=400, Stim_amount=30, t_list=[2.5, 1, 0.5, 0.25, 0.13]):
    """Recognizes different stimuli within the stimulation time series"""
    # threshold list for stimulus recognition,needs to be adjusted for different stimulation patterns
    Stim_types = np.zeros((Stim_amount, 2))
    Stimulus_diff = np.diff(Stimulus)
    Stim_count = 0
    dead = 0
    for ii in np.arange(deadtime, len(Stimulus_diff)):
        if Stimulus[ii] - Stimulus[ii - 1] > threshold and dead <= 0:
            Stim_types[Stim_count, 0] = ii

            if Stimulus[ii + 4] - Stimulus[ii - 1] > t_list[0]:
                Stim_types[Stim_count, 1] = 1
            elif Stimulus[ii + 4] - Stimulus[ii - 1] < t_list[0] and Stimulus[ii + 4] - Stimulus[ii - 1] > t_list[1]:
                Stim_types[Stim_count, 1] = 2
            elif Stimulus[ii + 4] - Stimulus[ii - 1] < t_list[1] and Stimulus[ii + 4] - Stimulus[ii - 1] > t_list[2]:
                Stim_types[Stim_count, 1] = 3
            elif Stimulus[ii + 4] - Stimulus[ii - 1] < t_list[2] and Stimulus[ii + 4] - Stimulus[ii - 1] > t_list[3]:
                Stim_types[Stim_count, 1] = 4
            elif Stimulus[ii + 4] - Stimulus[ii - 1] < t_list[3] and Stimulus[ii + 4] - Stimulus[ii - 1] > t_list[4]:
                Stim_types[Stim_count, 1] = 5
            elif Stimulus[ii + 4] - Stimulus[ii - 1] < t_list[4]:
                Stim_types[Stim_count, 1] = 6
            Stim_count = Stim_count + 1
            dead = deadtime
        else:
            dead = dead - 1

    return Stim_types


def computations(input_data):
    txtFile = input_data[0]
    new_dir = input_data[1]
    dirFile = input_data[2]
    if txtFile.endswith(".txt"):
        fnames = new_dir + txtFile  # filename to be processed

        tactFiles = os.listdir(dir_name + dirFile + '/')
        for tactFile in tactFiles:
            if txtFile[26] == '.':
                if tactFile.endswith('nr' + txtFile[-5] + '_tactile.txt') & tactFile.startswith(
                        txtFile[-12:-6]):
                    snames = dir_name + dirFile + '/' + tactFile
            elif txtFile[28] == '.':
                if tactFile.endswith(txtFile[-7:-4] + '_tactile.txt') & tactFile.startswith(txtFile[-14:-8]):
                    snames = dir_name + dirFile + '/' + tactFile

        # ++ Nisse ++ Pandas is 14x faster than numpy here:
        # Using R datatable is even a bit faster: 1,3 times ;D
        Signal = datatable.fread(fnames).to_numpy()
        Stimulus = datatable.fread(snames).to_numpy()

        frames = Signal.shape[0]
        # --- Stimulus Correction
        # Reducing framerate from 10000 to 1000 by taking every 10th value
        Stimulus = Stimulus[1::10, 3]
        # correcting for artefacts
        Stimulus[Stimulus < 2] = 2
        for ii in np.arange(1, len(Stimulus)):
            if Stimulus[ii] - Stimulus[ii - 1] < -0.1 and Stimulus[ii + 1] - Stimulus[ii] > 0.1:
                Stimulus[ii] = np.mean([Stimulus[ii - 1], Stimulus[ii + 1]])

        # Stimulus x-axis for plotting
        Stimulus_frame = np.arange(0, frames, frames / Stimulus.shape[0])

        # Gets stimulus information
        Stim_types = define_stimuli(Stimulus)

        Stim_types_fr = Stim_types
        Stim_types_fr[:, 0] = Stim_types[:, 0] * fr // fr_Stimulus
        # Sorts them
        Stim_types_fr_sorted = Stim_types_fr[Stim_types_fr[:, 1].argsort()]

        # --- Überdenken
        if np.max(Stim_types_fr_sorted[:, 0]) >= len(Signal):
            Signal = np.concatenate((Signal, np.zeros(
                (int(np.max(Stim_types_fr_sorted[:, 0]) - len(Signal) + time_window), Signal.shape[1]))))

        # Involves only Stimulus pattern 1
        if np.count_nonzero(Stim_types_fr_sorted == 3) == Stim_type_amount & np.count_nonzero(
                Stim_types_fr_sorted == 3) == Stim_type_amount:

            # --- DeltaF and Zscore calculation:
            base = np.percentile(Signal, 5, axis=0)
            DeltaF = (Signal - base) / (base)

            zscore_pre = stats.zscore(DeltaF)

            # --- DeltaF values
            box = np.ones(box_pts) / box_pts

            # Scales smoothed values
            zscore = np.zeros((zscore_pre.shape))
            for ii in np.arange(len(zscore)):
                # Roling average over DeltaF
                zscore[ii, :] = np.convolve(zscore_pre[ii, :], box, mode='same')

            # ---  Regression analysis
            time = np.arange(0, time_window / fr, 1 / fr)
            regressor = np.zeros((1, time_window))
            regressor[:, start + 1:start + 2] = 1

            # Convolution
            # CIRF: Calcium Impulse Response Function
            CIRF = np.exp(-time / tau)
            convolved = np.zeros((1, time_window))
            zeroPadded = np.zeros((time_window * 3))
            zeroPadded[time_window:time_window * 2] = regressor[0, :]
            convReg = np.convolve(zeroPadded, CIRF, 'full')
            convolved[0, :] = convReg[time_window:time_window * 2]
            # Gc is the CIRF at the desired time/sample point
            Gc = np.transpose(convolved)

            # Cutouts
            cutouts = np.zeros((Stim_amount * Stim_type_amount, Signal.shape[1], time_window))
            for kk in np.arange(len(Stim_types_fr_sorted)):
                for jj in np.arange(Signal.shape[1]):
                    cutouts[kk, jj, :] = zscore[int(Stim_types_fr_sorted[kk, 0] - start):int(
                        Stim_types_fr_sorted[kk, 0] + (time_window - start)), jj]

            # Linear regression and score calculation (with means)
            cutouts_reshape = np.reshape(cutouts, (cutouts.shape[0] * cutouts.shape[1], time_window))
            Scores = np.zeros((cutouts_reshape.shape[0], 1))

            for nn in np.arange(len(Scores)):
                reg = LinearRegression().fit(Gc, np.transpose(cutouts_reshape)[:, nn])
                r_squared = reg.score(Gc, np.transpose(cutouts_reshape)[:, nn])
                coef = reg.coef_
                Scores[nn, :] = coef * r_squared

            Scores_mean = np.mean(np.reshape(Scores, (Stim_amount, Stim_type_amount, Signal.shape[1])), axis=1)

            cell_response_index = []
            for nn in range(Signal.shape[1]):
                for rr in range(Stim_amount):
                    if Scores_mean[rr, nn] > score_treshold:
                        cell_response_index.append(nn)
                        break
            # Save Data to HDD
            decimal_precision = '18e'  # is this precision needed?
            # e takes a tiny bit longer than f notation ;D
            # Using 8f will cut the overall disc space used in half, 8mb vs 4mb, in this case its irrelevant ;D
            # This way is more pythonic an easier to read:
            file_export_path = f'{dir_name}results/{txtFile[:-4]}'
            np.savetxt(f'{file_export_path}_Scores.txt', np.transpose(Scores_mean[:, cell_response_index]),
                       fmt=f'%.{decimal_precision}')
            np.savetxt(f'{file_export_path}_cutouts.txt',
                       np.reshape(cutouts[:, cell_response_index, :],
                                  (cutouts.shape[0] * len(cell_response_index), time_window)),
                       fmt=f'%.{decimal_precision}')
            np.savetxt(f'{file_export_path}_cell_response_index.txt', cell_response_index,
                       fmt=f'%.{decimal_precision}')
            np.savetxt(f'{file_export_path}_zscore.txt', zscore[:, cell_response_index],
                       fmt=f'%.{decimal_precision}')

            print(cell_response_index)


########################################################################################################################
# SCRIPT ###############################################################################################################
# dir_name = 'C:/Users/nilsw/Documents/Dokumente Nils/Arbeit und Bewerbungen/Jobs/Biologie 1/Bio_Melanie/Data/Raw_data/'

dir_name = 'C:/Uni Freiburg/NilsWenke_Datenauswertung_2021/Raw_data/'
rawFiles = os.listdir(dir_name)
fr = 2.0345
fr_Stimulus = 100
tau = 3
time_window = 50
start = 10
Stim_type_amount = 5  # Amount of different stimulus types
Stim_amount = 6  # Repeats of one stimulus type
score_treshold = 0.5
box_pts = 3
time_it_took = []
if __name__ == '__main__':
    t1 = stop_watch.time()
    # Finds stimulus files and matching stimulation files
    for dirFile in rawFiles:
        if dirFile.endswith("DOB"):
            new_dir = dir_name + dirFile + '/Scripts and Results/'
            allFiles = os.listdir(new_dir)
            func_input = [allFiles, new_dir, dirFile]
            cpus = os.cpu_count() - 1
            with Pool(processes=cpus) as p:
                print(p.map(computations, func_input))
    t2 = stop_watch.perf_counter()
    print(t2 - t1)
    # In total this is now ca. 7 times faster!
    print('Finished!')
