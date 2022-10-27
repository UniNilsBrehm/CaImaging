import numpy as np
from IPython import embed
import tifftools
import os
import time
from tkinter.filedialog import askdirectory
from tkinter import Tk
import analysis_util_functions as uf
import pandas as pd


def suite2p_registering(rec_path):
    # How to load and read ops.npy files:
    #  np.load(p, allow_pickle=True).item()
    print('')
    print('-------- INFO --------')
    print('WILL IMPORT SUITE2P PACKAGE ... THIS MAY TAKE A FEW SECONDS ...')
    print('')
    import suite2p
    # original_tiff_file_name = 'original.TIF'
    # stimulus_file_name = 'stimulation.txt'

    t0 = time.time()
    # Check if selected directory is a correct one
    # dir_control = ['metadata.csv', 'logs', 'original', 'rawdata', 'references', 'stimulation', 'tiffs', 'stacks', 'figs', 'reg']
    # dir_control = ['metadata.csv', 'original', 'recording', 'stimulation', 'figs', 'reg', 'protocol']
    # check_dir_structure = uf.check_directory_structure(dir_path=rec_path, dir_control=dir_control)
    # if not check_dir_structure:
    #     exit()

    # Set directories
    # stimulation_dir = f'{rec_path}/stimulation/'
    # tiff_path = f'{rec_path}/original/'
    # store_path = f'{rec_path}/reg/'
    rec_name = os.path.split(rec_dir)[1]
    reg_suite2_path = f'{rec_path}/suite2p/plane0/reg_tif/'

    # Find tiff file and rename it
    file_list = os.listdir(rec_path)
    tif_file_name = [s for s in file_list if 'recording' in s]
    if len(tif_file_name) > 1:
        uf.msg_box('WARNING', 'Found more than one tif file!', '+')
    else:
        uf.msg_box('INFO', f'Found {tif_file_name}', '+')

    # Check if needed files are there:
    # Check if needed files are there:
    # check1 = uf.check_files(file_name=original_tiff_file_name, file_path=tiff_path)
    # check2 = uf.check_files(file_name=stimulus_file_name, file_path=stimulation_dir)
    # if not check1 * check2:
    #     exit()

    # Load metadata
    # metadata_df = pd.read_csv(f'{rec_path}/metadata.csv')

    # Settings:
    non_rigid_msg = input('Do Non-Rigid Registration? (y/n): ')
    if non_rigid_msg == 'y':
        non_rigid = True
        # metadata_df['RegistrationMethod'] = ['Suite2p-Non-Rigid']
        print('')
        print('---------- INFO ----------')
        print('Non-Rigid Registration selected!')
        print('')
    else:
        non_rigid = False
        # metadata_df['RegistrationMethod'] = ['Suite2p-Rigid']
        print('')
        print('---------- INFO ----------')
        print('Rigid Registration selected!')
        print('')

    # Store metadata to HDD
    # metadata_df.to_csv(f'{rec_path}/metadata.csv', index=False)

    ops = suite2p.default_ops()  # populates ops with the default options
    ops['tau'] = 1.25
    ops['fs'] = 2
    ops['nimg_init'] = 300  # (int, default: 200) how many frames to use to compute reference image for registration
    ops['batch_size'] = 300  # (int, default: 200) how many frames to register simultaneously in each batch. This depends
    # on memory constraints - it will be faster to run if the batch is larger, but it will require more RAM.

    ops['reg_tif'] = True  # store reg movie as tiff file
    ops['nonrigid'] = non_rigid  # (bool, default: True) whether or not to perform non-rigid registration,
    # which splits the field of view into blocks and computes registration offset in each block separately.

    ops['block_size'] = [128, 128]  # (two ints, default: [128,128]) size of blocks for non-rigid registration, in pixels.
    # HIGHLY recommend keeping this a power of 2 and/or 3 (e.g. 128, 256, 384, etc) for efficient fft

    ops['roidetect'] = False  # (bool, default: True) whether or not to run ROI detect and extraction

    db = {
        'data_path': [rec_path],
        'save_path0': rec_path,
        'tiff_list': tif_file_name,
        'subfolders': [],
        'fast_disk': rec_path,
        'look_one_level_down': False,
    }

    # Store suite2p settings
    pd.DataFrame(ops.items(), columns=['Parameter', 'Value']).to_csv(
        f'{rec_path}/{rec_name}_reg_settings_ops.csv', index=False)
    pd.DataFrame(db.items(), columns=['Parameter', 'Value']).to_csv(
        f'{rec_path}/{rec_name}_reg_settings_db.csv', index=False)

    # Run suite2p pipeline in terminal with the above settings
    output_ops = suite2p.run_s2p(ops=ops, db=db)
    print('')
    print('---------- REGISTRATION FINISHED ----------')
    print('')

    print('---------- COMBINING TIFF FILES ----------')
    print('')

    # Load registered tiff files
    f_list = sorted(os.listdir(reg_suite2_path))
    print('FOUND REGISTERED SINGLE TIFF FILES:')
    print(f_list)
    # Load first tiff file
    im_combined = tifftools.read_tiff(f'{reg_suite2_path}{f_list[0]}')

    # Combine tiff files to one file
    for k, v in enumerate(f_list):
        if k == 0:
            continue
        else:
            im_dummy = tifftools.read_tiff(f'{reg_suite2_path}{v}')
            im_combined['ifds'].extend(im_dummy['ifds'])
    # Store combined tiff file
    tifftools.write_tiff(im_combined, f'{rec_path}/{rec_name}_Registered.tif')
    t1 = time.time()
    print('----------------------------------------')
    print('++++++++++++++++++++++++++++++++++++++++')
    print('----------------------------------------')
    print('Stored Registered Tiff File to HDD')
    print(f'This took approx. {np.round((t1-t0)/60, 2)} min.')


if __name__ == '__main__':
    uf.msg_box('SUITE2P REGISTRATION', 'Please select directory containing one recording (tif) file', '+')
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    rec_dir = askdirectory()
    suite2p_registering(rec_path=rec_dir)
