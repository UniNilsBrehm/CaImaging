import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from tkinter.filedialog import askdirectory
from tkinter.filedialog import askopenfile
from tkinter import Tk
from IPython import embed
from scipy.signal import hilbert
import scipy.signal as sig
from sklearn.linear_model import LinearRegression
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster.vq import whiten
from sklearn.decomposition import PCA
import seaborn as sns


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def compute_clustering():
    msg_box('CLUSTERING', 'STARTING CLUSTERING', sep='-')
    Tk().withdraw()
    base_dir = askdirectory()
    file_list = os.listdir(base_dir)

    lm_scoring_file_name = [s for s in os.listdir(base_dir) if 'linear_model_stimulus_scoring' in s][0]
    lm_scoring_long = pd.read_csv(f'{base_dir}/{lm_scoring_file_name}')

    # Convert Long to wide format (needed for clustering)
    # lm_scoring = lm_scoring_long.pivot_table(index='roi', columns='stimulus_id', values='score')
    # stimulus_ids = list(lm_scoring.keys())

    # Average over trials (if stimulus type has multiple trials)
    lm_scoring_averaged = average_over_trials(data=lm_scoring_long)
    lm_scoring = lm_scoring_averaged.pivot_table(index='roi', columns='stimulus_id', values='score')
    # d = mean_lm_scoring.copy()
    stimulus_types = list(lm_scoring.keys())

    fig, axs = plt.subplots(5, 2)
    axs = axs.flatten()
    th = 0.2
    for k, ax in zip(stimulus_types, axs):
        idx = lm_scoring[k] >= th
        cells_above_th = idx.sum()

        ax.hist(lm_scoring[k], bins=int(len(lm_scoring[k])*0.2), density=False)
        # ax.hist(d[k], bins=len(d[k])//20, cumulative=-1, histtype='step', density=False)
        ax.set_title(f'{k}: {cells_above_th} >= {th}')
        ax.set_xlim(-0.5, 0.6)
    plt.tight_layout()
    plt.show()


    # standardize (normalize) the features
    # data = whiten(mean_lm_scoring)
    data = lm_scoring.copy()
    embed()
    exit()
    # COMPUTE CLUSTERING
    # Distance Metric available:
    # ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’,
    # ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’,
    # ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.
    # Methods available:
    # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

    # Compute the distance matrix
    matrix = linkage(
        data,
        method='ward',
        metric='euclidean',
        optimal_ordering=True
    )

    matrix = linkage(
        data,
        method='average',
        metric='correlation',
        optimal_ordering=True
    )

    # Plot Dendrogram
    # default color threshold: 0.7 * np.max(matrix[:, 2])
    plt.figure()
    dn = dendrogram(matrix, truncate_mode="level", p=0, distance_sort=True, show_leaf_counts=True,
                    color_threshold=0.3 * np.max(matrix[:, 2]))
    plt.title('Dendrogram2')

    plt.show()

    # --------------------------------------
    # Assign cluster labels (Stimulus IDs)
    labels = fcluster(
        matrix, t=5,
        criterion='maxclust'
    )

    # Create DataFrame for Scatter Plot
    # df = pd.DataFrame(data, columns=stimulus_ids)
    # df['labels'] = labels
    data_plotting = data.copy()
    data_plotting['labels'] = labels
    # Plot Clusters

    # sns.scatterplot(
    #     x='looming',
    #     y='grating_180',
    #     hue='labels',
    #     data=data_plotting
    # )
    # plt.show()

    x = 'grating_180'
    y = 'looming'
    z = 'movingtargetlarge'
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data_plotting[x], data_plotting[y], data_plotting[z], c=labels, cmap='viridis')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    plt.show()

    # --------
    # PCA of the Scores to reduce dimensions (number of resulting clusters)
    pca = PCA(n_components=3)
    pca.fit(data.T)
    explained_variance_ratio = pca.explained_variance_ratio_
    singular_values = pca.singular_values_
    components = pca.components_
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(components[0], components[1], components[2])
    ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.2f})')
    ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.2f})')
    ax.set_zlabel(f'PC3 ({explained_variance_ratio[2]:.2f})')
    plt.show()

    # ----------------------------------
    # Seaborn has a plotting function that includes a hierarchically-clustered heatmap (based on scipy)
    sns.clustermap(data.T, method='ward')
    plt.show()

    sns.clustermap(data.T, z_score=1)
    sns.clustermap(data.T, standard_scale=1)
    plt.show()

    # Standardize or Normalize every column in the figure
    # Standardize:
    sns.clustermap(lm_scoring, standard_scale=0)
    plt.show(
    )
    # Normalize
    sns.clustermap(lm_scoring, z_score=0)
    plt.show()

    # use the outlier detection
    sns.clustermap(lm_scoring, robust=True)
    plt.show()

    # SK LEARN PACKAGE
    # # X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    # model = AgglomerativeClustering(n_clusters=10)
    model = model.fit(data)

    plt.figure()
    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode="level", p=4, distance_sort=True, show_leaf_counts=True)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

    # NOTES
    # The FeatureAgglomeration uses agglomerative clustering to group together features that look very similar,
    # thus decreasing the number of features. It is a dimensionality reduction tool.
    # https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering

    # K-Means Clustering
    kmeans_model = KMeans(n_clusters=10, random_state=1).fit(data)
    labels = kmeans_model.labels_
    metrics.silhouette_score(data, labels, metric='euclidean')

    s = []
    for n in range(2, 20):
        kmeans_model = KMeans(n_clusters=n, random_state=1).fit(data)
        labels = kmeans_model.labels_
        s.append(metrics.silhouette_score(data, labels, metric='euclidean'))


def delta_f_over_f(data, p):
    fbs = np.percentile(data, p, axis=0)
    df = (data - fbs) / fbs
    return df


def msg_box(f_header, f_msg, sep, r=30):
    print(f'{sep * r} {f_header} {sep * r}')
    print(f'{sep * 2} {f_msg}')
    print(f'{sep * r}{sep}{sep * len(f_header)}{sep}{sep * r}')


def z_transform(data):
    result = (data - np.mean(data)) / np.std(data)
    return result


def moving_average_filter(data, window):
    return np.convolve(data, np.ones(window) / window, mode='same')


def envelope(data, rate, freq=100.0):
    # Low pass filter the absolute values of the signal in both forward and reverse directions,
    # resulting in zero-phase filtering.
    sos = sig.butter(2, freq, 'lowpass', fs=rate, output='sos')
    filtered = sig.sosfiltfilt(sos, data)
    env = np.sqrt(2)*sig.sosfiltfilt(sos, np.abs(data))
    return filtered, env


def low_pass_filter(data, rate, freq=100.0):
    sos = sig.butter(2, freq, 'lowpass', fs=rate, output='sos')
    filtered = sig.sosfiltfilt(sos, data)
    return filtered


def apply_linear_model(xx, yy, norm_reg=True):
    # Normalize data to [0, 1]
    if norm_reg:
        f_y = yy / np.max(yy)
    else:
        f_y = yy

    # Check dimensions of reg
    if xx.shape[0] == 0:
        print('ERROR: Wrong x input')
        return 0, 0, 0
    if yy.shape[0] == 0:
        print('ERROR: Wrong y input')
        return 0, 0, 0

    if len(xx.shape) == 1:
        reg_xx = xx.reshape(-1, 1)
    elif len(xx.shape) == 2:
        reg_xx = xx
    else:
        print('ERROR: Wrong x input')
        return 0, 0, 0

    # Linear Regression
    l_model = LinearRegression().fit(reg_xx, f_y)
    # Slope (y = a * x + c)
    a = l_model.coef_[0]
    # R**2 of model
    f_r_squared = l_model.score(reg_xx, f_y)
    # Score
    f_score = a * f_r_squared
    return f_score, f_r_squared, a


def convert_to_secs(data, method=0):
    if method == 0:
        # INPUT: hhmmss.ms (163417.4532)
        s = str(data)
        in_secs = int(s[:2]) * 3600 + int(s[2:4]) * 60 + float(s[4:])
    elif method == 1:
        # INPUT: hh:mm:ss
        s = data
        in_secs = int(s[:2]) * 3600 + int(s[3:5]) * 60 + float(s[6:])
    else:
        in_secs = None
    return in_secs


def visual_stimulation():
    """ Collect all (visual) stimulus text files and combine them into one data frame (csv file)

    Returns
    -------

    """
    msg_box('VISUAL STIMULATION', 'STARTING TO GET ALL VISUAL STIMULATION FILES', sep='-')
    Tk().withdraw()
    base_dir = askdirectory()
    stimulus_data_dir = f'{base_dir}/stimulation'

    # Load Meta Data
    file_list = os.listdir(base_dir)
    meta_data_file_name = [s for s in file_list if 'meta_data.csv' in s][0]
    meta_data = pd.read_csv(f'{base_dir}/{meta_data_file_name}')
    # time_zero = float(meta_data[meta_data['parameter'] == 'rec_vr_start']['value'].item())
    time_zero = float(meta_data[meta_data['parameter'] == 'rec_img_start_plus_delay']['value'].item())
    # rec_duration = float(meta_data[meta_data['parameter'] == 'rec_vr_duration']['value'].item())

    file_list = os.listdir(stimulus_data_dir)
    # vr_files = [s for s in file_list if "ephys" in s]
    stimulus = dict()
    for f_name in file_list:
        stimulus[f_name[6:-4]] = pd.read_csv(f'{stimulus_data_dir}/{f_name}', sep='\t', header=None)
    stimulus_id = file_list[0][:5]
    idx = list(stimulus.keys())
    stimulus_df = pd.DataFrame()
    for stim_name in idx:
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # MOVING TARGET SMALL
        if stim_name == 'movingtargetsmall':
            # convert all timestamps to secs
            t_secs = []
            for i in range(stimulus[stim_name].shape[0]):
                t_secs.append(convert_to_secs(stimulus[stim_name][3].iloc[i][11:], method=1) - time_zero)
            t_secs = np.array(t_secs)

            # Get start and end times of the trials (via the large pause in between)
            d = np.diff(t_secs, append=0)
            idx = np.where(d > 15)
            trial_start = [t_secs[0]]
            trial_end = []
            for i in idx[0]:
                trial_end.append(t_secs[i])
                trial_start.append(t_secs[i+1])
            trial_end.append(t_secs[-1])
            val = 1
            cc = 1
            # put all trials and their values into the main data frame
            # [onset, offset, binary value, stimulus id, stimulus type, info, trial]
            for s, e in zip(trial_start, trial_end):
                stimulus_df = pd.concat([stimulus_df, pd.Series([s, e, val, f'{stim_name}_{cc}', stim_name, 0, cc]).to_frame().T], ignore_index=True)
                cc += 1

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # MOVING TARGET Large
        if stim_name == 'movingtargetlarge':
            # convert all timestamps to secs
            t_secs = []
            for i in range(stimulus[stim_name].shape[0]):
                t_secs.append(convert_to_secs(stimulus[stim_name][3].iloc[i][11:], method=1) - time_zero)
            t_secs = np.array(t_secs)

            # Get start and end times of the trials (via the large pause in between)
            d = np.diff(t_secs, append=0)
            idx = np.where(d > 15)
            trial_start = [t_secs[0]]
            trial_end = []
            for i in idx[0]:
                trial_end.append(t_secs[i])
                trial_start.append(t_secs[i+1])
            trial_end.append(t_secs[-1])

            val = 2
            cc = 1
            # put all trials and their values into the main data frame
            # [onset, offset, binary value, stimulus id, stimulus type, info, trial]
            for s, e in zip(trial_start, trial_end):
                stimulus_df = pd.concat([stimulus_df, pd.Series([s, e, val, f'{stim_name}_{cc}', stim_name, 0, cc]).to_frame().T], ignore_index=True)
                cc += 1

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # FLASH
        if stim_name == 'flash':
            # convert all timestamps to secs
            t_secs = []
            for i in range(stimulus[stim_name].shape[0]):
                t_secs.append(convert_to_secs(stimulus[stim_name][1].iloc[i][11:], method=1) - time_zero)

            # t_secs = np.array(t_secs)
            t_secs = t_secs[3:]
            flash_dur = 4  # secs

            # The first entry is time when light turns on, then it turns off, then on etc.
            light_on_off = t_secs[::2]
            light_on_off.append(t_secs[-1])
            cc = 1
            cc_on = 0
            cc_off = 0
            for i in light_on_off:
                if (cc % 2) == 0:
                    state = 'OFF'
                    cc_off += 1
                    val = -1
                    cc_trial = cc_off
                else:
                    state = 'ON'
                    cc_on += 1
                    val = 1
                    cc_trial = cc_on
                stimulus_df = pd.concat(
                    [stimulus_df,
                     pd.Series(
                         [i, i+flash_dur, val, f'{stim_name}_light_{cc_trial}_{state}', stim_name, state, cc_trial]).to_frame().T],
                    ignore_index=True)
                cc += 1

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # GRATINGS
        if stim_name == 'grating':
            orientation = [0, 90, 180, 270]
            vals = [0.25, 0.5, 0.75, 1.0]
            # start_sec = []
            # end_sec = []
            cc = 0
            for o in orientation:
                idx_o = stimulus[stim_name].iloc[:, 1] == o
                s = stimulus[stim_name][idx_o][2].iloc[0][11:]
                # start_sec.append(convert_to_secs(s, method=1) - time_zero)
                start_sec = convert_to_secs(s, method=1) - time_zero
                e = stimulus[stim_name][idx_o][2].iloc[-1][11:]
                end_sec = convert_to_secs(e, method=1) - time_zero
                stimulus_df = pd.concat([stimulus_df, pd.Series([start_sec, end_sec, vals[cc], f'{stim_name}_{o}', stim_name, o, 1]).to_frame().T], ignore_index=True)
                cc += 1

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # LOOMING
        if stim_name == 'looming':
            val = 1
            s = stimulus[stim_name][3].iloc[0][11:]
            e = stimulus[stim_name][3].iloc[-1][11:]
            start_sec = convert_to_secs(s, method=1) - time_zero
            end_sec = convert_to_secs(e, method=1) - time_zero
            stimulus_df = pd.concat([stimulus_df, pd.Series([start_sec, end_sec, val, stim_name, stim_name, 0, 1]).to_frame().T], ignore_index=True)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # REVERSE LOOMING
        if stim_name == 'looming_rev':
            val = -1
            s = stimulus[stim_name][3].iloc[0][11:]
            e = stimulus[stim_name][3].iloc[-1][11:]
            start_sec = convert_to_secs(s, method=1) - time_zero
            end_sec = convert_to_secs(e, method=1) - time_zero
            stimulus_df = pd.concat([stimulus_df, pd.Series([start_sec, end_sec, val, stim_name, stim_name, 1, 1]).to_frame().T], ignore_index=True)

    # Names of the data frame columns
    # [onset, offset, binary value, stimulus id, stimulus type, info, trial]
    stimulus_df.columns = ['start', 'end', 'value', 'stimulus_id', 'stimulus_type', 'info', 'trial']

    # sort data by onset time (start)
    stimulus_df_sorted = stimulus_df.sort_values(by=['start'])
    print(stimulus_df_sorted)

    # Store to HDD
    stimulus_df_sorted.to_csv(f'{base_dir}/stimulus_protocol.csv', index=False)
    print('Stimulus Protocol store to HDD')


def create_binary(stimulus_protocol, ca_fr, ca_duration):
    """ Creates Binaries for each stimulus type and converts onset and offset times into time points that fit into the
    Ca Imaging Recording (frame rate).

    Parameters
    ----------
    stimulus_protocol: must be a pandas dataframe
    ca_fr: float
    ca_duration: float

    Returns
    -------
    stimulus_binaries: binary traces for each stimulus type
    stimulus_time_points: pandas data frame with onset and offset times plus metadata

    """
    ca_time = np.arange(0, ca_duration, 1/ca_fr)
    stimulus_binaries = dict()
    stimulus_times = []
    for k in range(stimulus_protocol.shape[0]):
        binary = np.zeros_like(ca_time)
        start = stimulus_protocol.iloc[k]['start']
        end = stimulus_protocol.iloc[k]['end']
        stimulus_type_id = stimulus_protocol.iloc[k]['stimulus_id']
        # Look where in the ca recording time axis is the stimulus onset
        idx_start = np.where(ca_time <= start)[0][-1] + 1
        idx_end = np.where(ca_time <= end)[0][-1] + 1
        binary[idx_start:idx_end] = 1
        stimulus_binaries[stimulus_type_id] = binary
        stimulus_times.append(
            [stimulus_type_id, stimulus_protocol.iloc[k]['stimulus_type'], stimulus_protocol.iloc[k]['trial'], stimulus_protocol.iloc[k]['info'],
             ca_time[idx_start], ca_time[idx_end], idx_start, idx_end])
    stimulus_times_points = pd.DataFrame(
        stimulus_times, columns=['stimulus_id', 'stimulus_type', 'trial', 'info', 'start', 'end', 'start_idx', 'end_idx'])
    return stimulus_binaries, stimulus_times_points


def get_meta_data():
    msg_box('GET META DATA', 'STARTING TO GET META DATA', sep='-')
    olympus_setup_delay = 0.680  # Ahsan's Value
    time_stamp_key = '[Acquisition Parameters Common] ImageCaputreDate'
    time_stamp_key_ms = '[Acquisition Parameters Common] ImageCaputreDate+MilliSec'
    rec_duration_key = 'Time Per Series'
    rec_dt_key = 'Time Per Frame'
    Tk().withdraw()
    file_dir = askdirectory()
    img_rec_meta_data_file = [s for s in os.listdir(file_dir) if 'recording_file_metadata' in s]
    if not img_rec_meta_data_file:
        print('COULD NOT FINDE IMAGE RECORDING META DATA FILE')
        print('Its file name must contain: "recording_file_metadata"')
        return None
    img_rec_meta_data_file = img_rec_meta_data_file[0]
    rec_meta_data = pd.DataFrame()
    if img_rec_meta_data_file.endswith('.csv'):
        rec_meta_data = pd.read_csv(f'{file_dir}/{img_rec_meta_data_file}', sep=',')
    elif img_rec_meta_data_file.endswith('.txt'):
        rec_meta_data = pd.read_csv(f'{file_dir}/{img_rec_meta_data_file}', sep='\t')
    else:
        print('ERROR: METADATA FILE HAS WRONG FORMAT!')
        return None

    print('')
    print(f'INFO: MANUALLY ADDED OLYMPUS SETUP DELAY of {olympus_setup_delay} s')
    print('')

    rec_name = img_rec_meta_data_file[:11]
    time_stamp = rec_meta_data[rec_meta_data['Key'] == time_stamp_key]['Value'].values[0][12:-1]
    time_stamp_ms = rec_meta_data[rec_meta_data['Key'] == time_stamp_key_ms]['Value'].values[0]
    img_rec_start = int(time_stamp[:2]) * 3600 + int(time_stamp[3:5]) * 60 + int(time_stamp[6:]) + int(time_stamp_ms)/1000
    img_rec_duration = float(rec_meta_data[rec_meta_data['Key'] == rec_duration_key]['Value'].item()) / (1000*1000)
    img_rec_dt = float(rec_meta_data[rec_meta_data['Key'] == rec_dt_key]['Value'].item()) / (1000*1000)
    img_rec_fr = 1/img_rec_dt

    raw_data_dir = f'{file_dir}/rawdata'
    file_list = os.listdir(raw_data_dir)
    # Get first time stamp of first vr file
    vr_files = [s for s in file_list if "ephys" in s]
    vr_files = list(np.sort(vr_files))
    print('FOUND FOLLOWING VENTRAL ROOT RECORDING FILES (sorted):')
    for i in vr_files:
        print(i)
    print('')

    vr_first = pd.read_csv(f'{raw_data_dir}/{vr_files[0]}', sep='\t', header=None)
    vr_last = pd.read_csv(f'{raw_data_dir}/{vr_files[-1]}', sep='\t', header=None)

    vr_start = convert_to_secs(vr_first[3].iloc[0])
    vr_end = convert_to_secs(vr_last[3].iloc[-1])
    print(f'VR RECORDING DURATION: {vr_end-vr_start:.3f} secs')

    # Meta Data File
    parameters_dict = {
        'rec_id': rec_name,
        'rec_img_start': img_rec_start,
        'rec_img_start_plus_delay': img_rec_start + olympus_setup_delay,
        'rec_img_duration': img_rec_duration,
        'rec_img_dt': img_rec_dt,
        'rec_img_fr': img_rec_fr,
        'rec_vr_start': vr_start,
        'rec_vr_end': vr_end,
        'rec_vr_duration': vr_end-vr_start
    }
    meta_data = pd.DataFrame(list(parameters_dict.items()), columns=['parameter', 'value'])
    meta_data.to_csv(f'{file_dir}/{rec_name}_meta_data.csv')
    print('Meta Data stored to HDD')
    print('')


def create_cif_double_tau(fr, tau1, tau2, a=1, t_max_factor=10):
    # Double Exponential Function to generate a Calcium Impulse Response Function
    # fr: Hz
    # tau1 and tau2: secs
    t_max = tau2 * t_max_factor  # in sec
    t_cif = np.arange(0, t_max*fr, 1)
    tau1_samples = tau1 * fr
    tau2_samples = tau2 * fr

    cif = a * (1 - np.exp(-(t_cif/tau1_samples)))*np.exp(-(t_cif/tau2_samples))
    return cif


def create_regressors():
    msg_box('CREATE REGRESSORS', 'STARTING TO COMPUTE STIMULUS REGRESSORS', sep='-')
    Tk().withdraw()
    base_dir = askdirectory()
    meta_data_file_name = [s for s in os.listdir(base_dir) if 'meta_data.csv' in s][0]
    binaries_file_name = [s for s in os.listdir(base_dir) if 'binaries' in s][0]

    meta_data = pd.read_csv(f'{base_dir}/{meta_data_file_name}')
    binaries = pd.read_csv(f'{base_dir}/{binaries_file_name}')

    ca_recording_fr = float(meta_data[meta_data['parameter'] == 'rec_img_fr']['value'])
    ca_impulse_response_function = create_cif_double_tau(fr=ca_recording_fr, tau1=0.1, tau2=1.0)

    # Convolve Binary with CIF to compute the regressors
    regs = dict()
    for k in range(binaries.shape[1]):
        regs[binaries.keys()[k]] = np.convolve(binaries.iloc[:, k], ca_impulse_response_function, 'full')
    pd.DataFrame(regs).to_csv(f'{base_dir}/stimulus_regressors.csv', index=False)
    print('REGRESSORS STORED TO HDD')


def export_binaries():
    msg_box('COMPUTE STIMULUS BINARIES', 'STARTING TO COMPUTE STIMULUS BINARIES', sep='-')
    Tk().withdraw()
    base_dir = askdirectory()
    meta_data_file_name = [s for s in os.listdir(base_dir) if 'meta_data.csv' in s][0]
    protocol_file_name = [s for s in os.listdir(base_dir) if 'protocol' in s][0]

    meta_data = pd.read_csv(f'{base_dir}/{meta_data_file_name}')
    protocol = pd.read_csv(f'{base_dir}/{protocol_file_name}')
    ca_recording_duration = float(meta_data[meta_data['parameter'] == 'rec_img_duration']['value'])
    ca_recording_fr = float(meta_data[meta_data['parameter'] == 'rec_img_fr']['value'])
    ca_time_axis = np.arange(0, ca_recording_duration, 1/ca_recording_fr)

    binaries, stimulus_times = create_binary(protocol, ca_recording_fr, ca_recording_duration)
    binaries_df = pd.DataFrame(binaries)
    all_stimuli_binary_trace = pd.DataFrame()
    all_stimuli_binary_trace['Time'] = ca_time_axis
    all_stimuli_binary_trace['Volt'] = binaries_df.sum(axis=1)

    # Store to HDD
    binaries_df.to_csv(f'{base_dir}/stimulus_binaries.csv', index=False)
    stimulus_times.to_csv(f'{base_dir}/stimulus_times.csv', index=False)
    all_stimuli_binary_trace.to_csv(f'{base_dir}/stimulus_trace.csv', index=False)
    print('Stimulus Binaries and Stimulus Times (in Ca Recording Frame Rate) stored to HDD')


def reg_analysis_cut_out_responses():
    """ Regressor-Analysis between cut out responses (ca imaging of rois) and stimulus regressors (binaries)

    Following files will be stored to HDD in the same directory as the input data:
    linear_model_stimulus_scoring.csv

    """
    # ToDo: - More information into doc string

    msg_box('REGRESSOR ANALYSIS', 'STARTING TO CUT OUT RESPONSES AND COMPUTE REGRESSORS AND LINEAR MODEL SCORING', sep='-')

    # padding in secs
    pad_before = 5
    pad_after = 10
    ca_dynamics_rise = 0.2
    shifting_limit = 3

    Tk().withdraw()
    base_dir = askdirectory()
    meta_data_file_name = [s for s in os.listdir(base_dir) if 'meta_data.csv' in s][0]
    # protocol_file_name = [s for s in os.listdir(base_dir) if 'stimulus_times' in s][0]
    protocol_file_name = [s for s in os.listdir(base_dir) if 'stimulus_protocol' in s][0]

    # regs_file_name = [s for s in os.listdir(base_dir) if 'regressor' in s][0]
    ca_rec_file_name = [s for s in os.listdir(base_dir) if 'raw_values' in s][0]

    meta_data = pd.read_csv(f'{base_dir}/{meta_data_file_name}')
    stimulus_protocol = pd.read_csv(f'{base_dir}/{protocol_file_name}')
    # regs = pd.read_csv(f'{base_dir}/{regs_file_name}')
    ca_rec_raw = pd.read_csv(f'{base_dir}/{ca_rec_file_name}', index_col=0)
    ca_recording_fr = float(meta_data[meta_data['parameter'] == 'rec_img_fr']['value'])

    # Convert raw values to delta f over f
    fbs_percentile = 5
    ca_rec = delta_f_over_f(ca_rec_raw, fbs_percentile)

    # Convert all parameters from secs to samples
    pad_before_samples = int(ca_recording_fr * pad_before)
    pad_after_samples = int(ca_recording_fr * pad_after)
    ca_dynamics_rise_samples = int(ca_recording_fr * ca_dynamics_rise)
    shifting_limit_samples = int(ca_recording_fr * shifting_limit)

    ca_fr = float(meta_data[meta_data['parameter'] == 'rec_img_fr']['value'])
    ca_duration = float(meta_data[meta_data['parameter'] == 'rec_img_duration']['value'])

    _, protocol = create_binary(stimulus_protocol, ca_fr, ca_duration)

    # Get unique stimulus types
    s_types = protocol['stimulus_type'].unique()

    # Loop through all ROIs (cols of the ca recording data frame)
    roi_scores = dict()
    result_list = []
    for roi_name in ca_rec:
        stimulus_type_scores = dict()
        # Loop through stimulus types (e.g. "movingtargetsmall")
        for s in s_types:
            # Get stimulus type
            p = protocol[protocol['stimulus_type'] == s]
            cc = []
            for k in range(p.shape[0]):
                # Get stimulus id (name)
                stimulus_id = p.iloc[k]['stimulus_id']
                stimulus_type = p.iloc[k]['stimulus_type']
                stimulus_trial = p.iloc[k]['trial']
                stimulus_info = p.iloc[k]['info']

                start = p.iloc[k]['start_idx'] - pad_before_samples
                end = p.iloc[k]['end_idx'] + pad_after_samples
                # single_stimulus_reg = regs[p['stimulus']].iloc[:, k].to_numpy()[start:end]
                ca_rec_cut_out = ca_rec.iloc[start:end].reset_index(drop=True)
                cc.append(ca_rec_cut_out)
                ca_trace = ca_rec_cut_out[roi_name].to_numpy()
                # Normalize (raw) trace to max = 1: For the final version there should be the delta f over f values
                ca_trace = ca_trace / np.max(ca_trace)

                # Create the Regressor on the fly
                binary = np.zeros_like(ca_trace)
                binary[pad_before_samples:len(binary)-pad_after_samples] = 1
                cif = create_cif_double_tau(ca_recording_fr, tau1=0.5, tau2=2.0)
                reg = np.convolve(binary, cif, 'full')
                reg = reg / np.max(reg)
                # ca_trace = reg + 0.4 * np.random.randn(len(reg))
                conv_pad = (len(reg) - len(ca_trace))
                # reg_same = reg[:-conv_pad]

                cross_corr_func = sig.correlate(ca_trace, reg, mode='full')
                lags = sig.correlation_lags(len(ca_trace), len(reg))
                # Get the lag time for the maximum cross corr value (in samples)
                cross_corr_optimal_lag = lags[np.where(cross_corr_func == np.max(cross_corr_func))[0][0]]

                # the response should not be before the stimulus and not too long after stimulus onset:
                if cross_corr_optimal_lag < 0:
                    cross_corr_optimal_lag = 0
                if cross_corr_optimal_lag > shifting_limit_samples:
                    cross_corr_optimal_lag = shifting_limit_samples

                # Consider Ca Rise Time
                cross_corr_optimal_lag = (cross_corr_optimal_lag - ca_dynamics_rise_samples)

                # Create regressor with optimal lag
                binary_optimal = np.zeros_like(ca_trace)
                binary_optimal[pad_before_samples + cross_corr_optimal_lag:len(binary_optimal) - pad_after_samples + cross_corr_optimal_lag] = 1
                reg_optimal = np.convolve(binary_optimal, cif, 'full')
                reg_optimal = reg_optimal / np.max(reg_optimal)
                conv_pad = (len(reg_optimal) - len(ca_trace))
                reg_same = reg_optimal[:-conv_pad]

                # Use Linear Regression Model to compute a Score
                sc, rr, aa = apply_linear_model(xx=reg_same, yy=ca_trace, norm_reg=True)
                msg = f"{roi_name}-{stimulus_id}: {sc:.3f}/{cross_corr_optimal_lag/ca_recording_fr:.3f} s"
                print(msg)

                # Prepare data frame entry
                cross_corr_optimal_lag_secs = cross_corr_optimal_lag / ca_recording_fr
                entry = [roi_name, stimulus_id, stimulus_type, stimulus_trial, stimulus_info, sc, cross_corr_optimal_lag_secs]
                result_list.append(entry)

                # # Test Plot
                # plt.figure()
                # plt.title(msg)
                # plt.plot(binary, 'b--')
                # plt.plot(binary_optimal, 'r--')
                # plt.plot(reg, 'b')
                # plt.plot(reg_optimal, 'r')
                # plt.plot(ca_trace, 'k')
                # plt.plot(reg_same, 'y--')
                # # plt.plot(cross_corr_func/np.max(cross_corr_func), 'g')
                # plt.show()

                # Collect Results for this stimulus type
                stimulus_type_scores[stimulus_id] = [sc, cross_corr_optimal_lag_secs]
        roi_scores[roi_name] = stimulus_type_scores
    # Put all results into one data frame (long format)
    results = pd.DataFrame(result_list, columns=['roi', 'stimulus_id', 'stimulus_type', 'trial', 'info', 'score', 'lag'])
    results.to_csv(f'{base_dir}/linear_model_stimulus_scoring.csv', index=False)
    print('Linear Model Scoring Results Stored to HDD!')


def reg_analysis_ventral_root():
    """ Regressor-Analysis between cut out responses (ca imaging of rois) and motor activity (ventral root recording)

    Following files will be stored to HDD in the same directory as the input data:
    linear_model_ventral_root_scoring.csv

    """
    # ToDo: - More information into doc string

    msg_box('REGRESSOR ANALYSIS', 'STARTING TO CUT OUT RESPONSES AND COMPUTE REGRESSORS AND LINEAR MODEL SCORING', sep='-')

    # padding in secs
    pad_before = 5
    pad_after = 10
    ca_dynamics_rise = 0.2
    shifting_limit = 3

    Tk().withdraw()
    base_dir = askdirectory()
    meta_data_file_name = [s for s in os.listdir(base_dir) if 'meta_data.csv' in s][0]
    binary_file_name = [s for s in os.listdir(base_dir) if 'ventral_root_binary' in s][0]
    ca_rec_file_name = [s for s in os.listdir(base_dir) if 'raw_values' in s][0]
    reg_file_name = [s for s in os.listdir(base_dir) if 'reg_trace' in s][0]

    meta_data = pd.read_csv(f'{base_dir}/{meta_data_file_name}')
    binary = pd.read_csv(f'{base_dir}/{binary_file_name}')
    reg = pd.read_csv(f'{base_dir}/{reg_file_name}')
    ca_rec_raw = pd.read_csv(f'{base_dir}/{ca_rec_file_name}', index_col=0)
    ca_recording_fr = float(meta_data[meta_data['parameter'] == 'rec_img_fr']['value'])

    # Convert raw values to delta f over f
    fbs_percentile = 5
    ca_rec = delta_f_over_f(ca_rec_raw, fbs_percentile)
    ca_fr = float(meta_data[meta_data['parameter'] == 'rec_img_fr']['value'])
    ca_duration = float(meta_data[meta_data['parameter'] == 'rec_img_duration']['value'])
    ca_time = np.arange(0, ca_duration, 1/ca_fr)

    rois = list(ca_rec.keys())
    results = []
    for roi_name in rois:
        # sc, rr, aa = apply_linear_model(xx=reg['Volt'].to_numpy(), yy=ca_rec[roi_name].to_numpy(), norm_reg=True)
        sc, rr, aa = apply_linear_model(xx=reg['Volt'].to_numpy(), yy=ca_rec[roi_name].to_numpy(), norm_reg=True)

        results.append([roi_name, sc, rr, aa])
    results_df = pd.DataFrame(results, columns=['roi', 'score', 'r_squared', 'slope'])

    plt.figure()
    plt.plot(ca_time, ca_rec['Mean1'], 'k')
    plt.plot(ca_time, reg['Volt'], 'b')
    plt.plot(ca_time, binary['Volt'], 'g')
    plt.show()

    embed()
    exit()

    # Convert all parameters from secs to samples
    pad_before_samples = int(ca_recording_fr * pad_before)
    pad_after_samples = int(ca_recording_fr * pad_after)
    ca_dynamics_rise_samples = int(ca_recording_fr * ca_dynamics_rise)
    shifting_limit_samples = int(ca_recording_fr * shifting_limit)


    # Compute Regressor Trace
    cif = create_cif_double_tau(fr=ca_fr, tau1=0.1, tau2=1)
    reg = np.convolve(binary['Volt'].to_numpy(), cif, 'full')
    conv_pad = (len(reg) - ca_rec.shape[0])
    reg_same = reg[:-conv_pad]
    rois = list(ca_rec.keys())
    results = []
    for roi_name in rois:
        sc, rr, aa = apply_linear_model(xx=reg_same, yy=ca_rec[roi_name].to_numpy(), norm_reg=True)
        results.append([roi_name, sc, rr, aa])
    results_df = pd.DataFrame(results, columns=['roi', 'score', 'r_squared', 'slope'])

    plt.figure()
    plt.plot(ca_time, ca_rec['Mean1'], 'k')
    plt.plot(ca_time, reg_same, 'b')
    plt.plot(ca_time, binary['Volt'], 'g')
    plt.show()

    reg_df = pd.DataFrame()
    reg_df['Time'] = ca_time
    reg_df['Volt'] = reg_same
    reg_df.to_csv(f'{base_dir}/ventral_root_reg_trace.csv', index=False)
    embed()
    exit()


def average_over_trials(data):
    """ Average over trials for "movingtargetsmall" and "movingtargetlarge", as well as for "flash ON" and "flash OFF".
    All the other stimulus types just copy from the original data frame.

    Parameters
    ----------
    data

    Returns
    -------

    """
    roi_names = data['roi'].unique()
    stimulus_types = data['stimulus_type'].unique()
    new_data = []
    for roi in roi_names:
        for s_name in stimulus_types:
            if (s_name == 'movingtargetsmall') or (s_name == 'movingtargetlarge'):
                idx = (data['stimulus_type'] == s_name) * (data['roi'] == roi)
                m_score = data[idx]['score'].mean()
                m_lag = data[idx]['lag'].mean()
                new_entry = [roi, s_name, s_name, 0, m_score, m_lag]
                new_data.append(new_entry)
            if s_name == 'flash':
                idx = (data['stimulus_type'] == s_name) * (data['roi'] == roi)
                dummy = data[idx]
                # get ON and OFF Flashs
                on_mean = dummy[dummy['info'] == 'ON']['score'].mean()
                on_lag = dummy[dummy['info'] == 'ON']['lag'].mean()
                off_mean = dummy[dummy['info'] == 'OFF']['score'].mean()
                off_lag = dummy[dummy['info'] == 'OFF']['lag'].mean()

                new_entry_on = [roi, f'{s_name}_ON', s_name, 'ON', on_mean, on_lag]
                new_entry_off = [roi, f'{s_name}_OFF', s_name, 'OFF', off_mean, off_lag]
                new_data.append(new_entry_on)
                new_data.append(new_entry_off)
    mean_scores_df = pd.DataFrame(new_data, columns=['roi', 'stimulus_id', 'stimulus_type', 'info', 'score', 'lag'])
    # Now add the other original entries that do not need averaging
    looms = data[data['stimulus_type'] == 'looming']
    looms_rev = data[data['stimulus_type'] == 'looming_rev']
    gratings = data[data['stimulus_type'] == 'grating']
    results = pd.concat([mean_scores_df, looms, looms_rev, gratings]).drop(columns='trial')
    results = results.sort_values(by='roi').reset_index(drop=True)

    # roi_names = data['roi'].unique()
    # stimulus_types = data['stimulus_type'].unique()
    # new_data = []
    # embed()
    # exit()
    # for roi in roi_names:
    #     for s_name in stimulus_types:
    #         idx = (data['stimulus_type'] == s_name) * (data['roi'] == roi)
    #         m_score = data[idx]['score'].mean()
    #         new_entry = [roi, s_name, m_score]
    #         new_data.append(new_entry)
    # mean_scores_df = pd.DataFrame(new_data, columns=['roi', 'stimulus_type', 'score'])

    return results


def transform_ventral_root_recording():
    msg_box('CONVERT RAW VENTRAL ROOT RECORDINGS', 'STARTING TO CONVERT ALL VENTRAL ROOT RECORDING FILES', sep='-')
    Tk().withdraw()
    base_dir = askdirectory()
    raw_data_dir = f'{base_dir}/rawdata'
    file_list = os.listdir(raw_data_dir)
    vr_files = [s for s in file_list if "ephys" in s]
    vr_files = list(np.sort(vr_files))
    # Load files
    vr_data = []
    vr_time_secs = []
    vr_multi_entries_values = []
    vr_multi_entries_count = []
    tag_val = 0
    cc = 0
    # Loop through every ventral root recording file
    t0 = time.perf_counter()
    for f_name in vr_files:
        vv = []
        print(f_name)
        dummy = pd.read_csv(f'{raw_data_dir}/{f_name}', sep='\t', header=None)
        vr_data.append(dummy)
        # Loop through every row in the ventral root file and convert the timestamp to secs
        for i, v in enumerate(dummy.iloc[:, 3].to_numpy()):
            s = convert_to_secs(v)
            vr_time_secs.append(s)
            # if i > 0:
            #     if s == vr_time_secs[i-1]:
            #         cc += 1
            #         vv.append(dummy.iloc[i, 0].item())
            #     else:
            #         vr_multi_entries_count.append(cc)
            #         if len(vv) == 0:
            #             print('WHAAAAT???')
            #         else:
            #             vr_multi_entries_values.append(np.mean(vv))
            #         cc = 0
            #         vv = []
            # else:
            #     cc += 1

            # if i > 0:
            #     if s == vr_time_secs[i-1]:
            #         vr_multi_entries_tag.append(tag_val)
            #     else:
            #         tag_val += 1
            #         vr_multi_entries_tag.append(tag_val)
            # else:
            #     vr_multi_entries_tag.append(tag_val)

    # Reset time so that it starts at 0
    vr_time_secs = np.array(vr_time_secs)
    vr_time_secs = vr_time_secs - vr_time_secs[0]

    # Concat all to one data frame
    print('... Concatenating all recordings into one data frame ...')
    vr_trace = pd.concat(vr_data).iloc[:, 0]

    # Correct for multiple entries for one time point
    start_here = 0
    vr_test = []
    for k, m_count in enumerate(vr_multi_entries_count):
        vr_test.append(vr_trace[start_here:m_count])
        start_here = m_count+1

    # Put all in one Data Frame
    vr_trace_export = pd.DataFrame(columns=['Time', 'Volt'])
    # Add the time in secs (not the timestamps)
    vr_trace_export['Time'] = vr_time_secs
    vr_trace_export['Volt'] = vr_trace.to_numpy()

    # Compute Envelope of VR Trace
    print('... Compute Envelope for Ventral Root Recording Trace ...')
    vr_fr = 10000
    vr_fil, vr_env = envelope(vr_trace_export['Volt'], vr_fr, freq=20.0)
    vr_env_export = pd.DataFrame(columns=['Time', 'Volt'])
    vr_env_export['Time'] = vr_trace_export['Time']
    vr_env_export['Volt'] = vr_env

    # Down-sample ventral root recording
    print('... Down-Sampling ...')
    ds_factor = 64
    vr_trace_export_ds = vr_trace_export[::ds_factor]
    vr_env_export_ds = vr_env_export[::ds_factor]

    # Export to HDD
    print('... Export Ventral Root Trace to HDD ...')
    vr_trace_export.to_csv(f'{base_dir}/ventral_root_trace.csv', index=False)
    vr_trace_export_ds.to_csv(f'{base_dir}/ventral_root_trace_ds_x{ds_factor}.csv', index=False)

    print('... Export Ventral Root Envelope to HDD ...')
    vr_env_export.to_csv(f'{base_dir}/ventral_root_envelope.csv', index=False)
    vr_env_export_ds.to_csv(f'{base_dir}/ventral_root_envelope_ds_x{ds_factor}.csv', index=False)
    t1 = time.perf_counter()
    print(f'Collecting all Recordings took: {(t1-t0):.2f} secs')


def ventral_root_detection():
    msg_box('VENTRAL ROOT ACTIVITY DETECTION', 'STARTING TO DETECT MOTOR EVENTS IN VENTRAL ROOT RECORDING', sep='-')
    Tk().withdraw()
    base_dir = askdirectory()
    file_list = os.listdir(base_dir)
    th = float(input('Enter Detection Threshold (SD): '))
    print(f'Threshold was set to {th} SD')

    meta_data_file_name = [s for s in os.listdir(base_dir) if 'meta_data.csv' in s][0]
    meta_data = pd.read_csv(f'{base_dir}/{meta_data_file_name}')
    vr_trace_file = [s for s in file_list if "ventral_root_trace" in s][0]
    vr_env_file = [s for s in file_list if "ventral_root_envelope" in s][0]
    vr_trace = pd.read_csv(f'{base_dir}/{vr_trace_file}')
    vr_env = pd.read_csv(f'{base_dir}/{vr_env_file}')
    ca_fr = float(meta_data[meta_data['parameter'] == 'rec_img_fr']['value'])
    ca_duration = float(meta_data[meta_data['parameter'] == 'rec_img_duration']['value'])
    ca_time = np.arange(0, ca_duration, 1/ca_fr)

    # Correct for multiple y values for one x value in ventral root recording
    # unique_times = vr_env['Time'].unique()
    # t = vr_env['Time'].to_numpy()
    # number_of_y_values = np.zeros_like(unique_times)
    # for k, ut in enumerate(unique_times):
    #     # print(f'{k} / {len(unique_times)}')
    #     idx = t == ut
    #     # number_of_y_values[k] = t[idx].shape[0]

    org_trace = vr_trace['Volt']
    # org = vr_env['Volt'] / np.max(vr_env['Volt'])
    org = vr_env['Volt']
    # Low pass filter ventral root envelope with a moving average
    org_fil = moving_average_filter(org, window=1000)
    # vr_z = z_transform(org_fil)
    vr_z = z_transform(org_fil)

    # Create Binary
    # th = 1  # threshold in SDs
    # th = low_pass_filter(vr_z, rate=1000, freq=300)
    binary = np.zeros_like(vr_z)
    binary[vr_z > th] = 1
    duration_th_secs = 5  # in secs
    duration_th = int(ca_fr * duration_th_secs)

    # Find onsets and offsets of ventral root activity
    onsets_offsets = np.diff(binary, append=0)
    time_axis = vr_env['Time']
    onset_idx = np.where(onsets_offsets > 0)[0]
    offset_idx = np.where(onsets_offsets < 0)[0]
    onset_times = time_axis.iloc[onset_idx]
    offset_times = time_axis.iloc[offset_idx]

    # check for motor activity that is too long (artifacts due to concatenating recordings) and remove it
    event_duration = offset_times.to_numpy() - onset_times.to_numpy()
    idx_remove = event_duration > duration_th_secs
    onset_times = onset_times[np.invert(idx_remove)]
    offset_times = offset_times[np.invert(idx_remove)]

    # Look where in the ca recording time axis is the stimulus onset time the closest to existing values
    vr_binary = np.zeros_like(ca_time)
    vr_activity = []
    vr_activity_onsets = np.zeros_like(onset_times)
    vr_activity_offsets = np.zeros_like(onset_times)
    for k in range(onset_times.shape[0]):
        start = onset_times.iloc[k]
        end = offset_times.iloc[k]
        idx_start = np.where(ca_time <= start)[0][-1] + 1
        idx_end = np.where(ca_time <= end)[0][-1] + 1

        # Make sure that each event ist at least one sample long
        if idx_start == idx_end:
            idx_end += 1

        if idx_end >= len(ca_time):
            idx_end = len(ca_time)
        if idx_start >= len(ca_time):
            continue

        time_start = ca_time[idx_start]
        time_end = ca_time[idx_end]
        dur = time_end - time_start
        # # check for motor activity that is too long (artifacts due to concatenating recordings)
        # if time_end - time_start > duration_th:
        #     print('FOUND TOO LONG MOTOR ACTIVITY EVENT AND REMOVED IT')
        #     continue
        vr_activity_onsets[k] = time_start
        vr_activity_offsets[k] = time_end
        vr_binary[idx_start:idx_end] = 1
        vr_activity.append([idx_start, idx_end, time_start, time_end, dur])
    # Test Reg
    cif = create_cif_double_tau(fr=ca_fr, tau1=0.5, tau2=3.0)
    reg_test, reg_test_same = reg_convolution(cif, vr_binary)
    reg_plot = (reg_test_same / np.max(reg_test_same)) * np.max(z_transform(org_trace))

    fig, axs = plt.subplots(2, 1, sharey=True, sharex=True)
    axs[0].plot(vr_trace['Time'], z_transform(org_trace), 'k')
    axs[0].plot(ca_time, reg_plot, 'r', lw=2)
    axs[1].plot(vr_trace['Time'], z_transform(org_trace), 'k', alpha=0.1)
    axs[1].plot(vr_env['Time'], vr_z, 'b', lw=1.5)
    # plt.plot(vr_env['Time'], z_transform(test), 'r')
    axs[1].plot(vr_env['Time'], binary * th, 'g', lw=2)
    axs[1].plot(vr_env['Time'], np.diff(binary, append=0)*10, 'r', lw=2)
    axs[1].plot(ca_time, vr_binary * th, 'go:', lw=1.5, alpha=0.7)
    axs[1].plot(ca_time, np.diff(vr_binary, append=0)*10, 'rx:', lw=1.5, alpha=0.7)
    plt.show()

    # Store to HDD
    vr_activity = pd.DataFrame(vr_activity, columns=['start_idx', 'end_idx', 'start_time', 'end_time', 'duration'])
    vr_activity.to_csv(f'{base_dir}/ventral_root_activity.csv', index=False)
    vr_binary_df = pd.DataFrame()
    vr_binary_df['Time'] = ca_time
    vr_binary_df['Volt'] = vr_binary
    vr_binary_df.to_csv(f'{base_dir}/ventral_root_binary_trace.csv', index=False)
    vr_reg_trace = pd.DataFrame()
    vr_reg_trace['Time'] = ca_time
    vr_reg_trace['Volt'] = reg_test_same
    vr_reg_trace.to_csv(f'{base_dir}/ventral_root_reg_trace.csv', index=False)
    print('Ventral Root Binary Stored to HDD')


def reg_convolution(cif, binary):
    reg = np.convolve(binary, cif, 'full')
    conv_pad = (len(reg) - binary.shape[0])
    reg_same = reg[:-conv_pad]
    return reg, reg_same


def print_options():
    print('')
    print('Type in the number of the function you want to use')
    print('')
    print('0: Export Meta data')
    print('1: Convert Ventral Root Recording')
    print('2: Export Stimulus Protocol')
    print('3: Export Stimulus Binaries (optional)' )
    print('4: Export Regressors (optional)')
    print('5: Regressor-based (LM-) Scoring')
    print('6: Ventral Root Activity Detection')
    print('7: Run Clustering')
    print('8: LM Scoring for Ventral Root')

    print('')
    print('To see options type: >> options')
    print('To exit type: >> exit')


if __name__ == '__main__':
    print_options()
    x = True
    while x:
        print('')
        usr = input("Enter: ")
        if usr == '0':
            get_meta_data()
        elif usr == '1':
            transform_ventral_root_recording()
        elif usr == '2':
            visual_stimulation()
        elif usr == '3':
            export_binaries()
        elif usr == '4':
            create_regressors()
        elif usr == '5':
            reg_analysis_cut_out_responses()
        elif usr == '6':
            ventral_root_detection()
        elif usr == '7':
            compute_clustering()
        elif usr == '8':
            reg_analysis_ventral_root()
        elif usr == 'options':
            print_options()
        elif usr == 'exit':
            exit()
