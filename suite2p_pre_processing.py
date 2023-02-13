import numpy as np
import tifftools
import os
import time
from tkinter.filedialog import askdirectory
from tkinter import Tk
import pandas as pd
import sys
"""
This Script can be used to do registration of calcium imaging recordings (tiff files) using the algorithms implemented
in the suite2p package. It will ask you to select a directory that includes only your tiff files. After that it will
automatically start the registration. If the recording file could be found it will ask you if you would like to do a
rigid or non-rigid registration.
If you want you can change the batch size by calling:
    >> python suite2p_pre_processing.py -500

Note:
    The file name must include "recording", otherwise it will be ignored!

Default settings:
    batch size of 300 frames
    frame rate: 2 Hz
    300 frames to use to compute reference image for registration
     
"""


def msg_box(f_header, f_msg, sep, r=30):
    print(f'{sep * r} {f_header} {sep * r}')
    print(f'{sep * 2} {f_msg}')
    print(f'{sep * r}{sep}{sep * len(f_header)}{sep}{sep * r}')


def suite2p_registering(rec_path, f_batch_size=300):
    # How to load and read ops.npy files:
    #  np.load(p, allow_pickle=True).item()

    t0 = time.time()
    # Set directories
    # stimulation_dir = f'{rec_path}/stimulation/'
    # tiff_path = f'{rec_path}/original/'
    # store_path = f'{rec_path}/reg/'
    rec_name = os.path.split(rec_dir)[1]
    reg_suite2_path = f'{rec_path}/suite2p/plane0/reg_tif/'

    # Find tiff file
    text_must_be_in_file_name = 'recording'
    file_list = os.listdir(rec_path)
    tif_file_name = [s for s in file_list if text_must_be_in_file_name in s]
    if len(tif_file_name) > 1:
        msg_box('WARNING', 'Found more than one tif file!', '+')
    else:
        if tif_file_name:
            msg_box('INFO', f'Found {tif_file_name}', '+')
        else:
            msg_box('ERROR', f'Found no tiff file!', '+')
            return

    print('')
    print('-------- INFO --------')
    print('WILL IMPORT SUITE2P PACKAGE ... THIS MAY TAKE A FEW SECONDS ...')
    print('')
    import suite2p
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
    ops['batch_size'] = f_batch_size  # (int, default: 200) how many frames to register simultaneously in each batch.
    # Depends on memory constraints - it will be faster to run if the batch is larger, but it will require more RAM.

    ops['reg_tif'] = True  # store reg movie as tiff file
    ops['nonrigid'] = non_rigid  # (bool, default: True) whether or not to perform non-rigid registration,
    # which splits the field of view into blocks and computes registration offset in each block separately.

    ops['block_size'] = [128, 128]  # (two ints, default: [128,128]) size of blocks for non-rigid reg, in pixels.
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
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command.startswith('-'):
            batch_size = int(command[1:])
        else:
            print('WRONG PARAMETER ... WIlL IGNORE IT')
            batch_size = 300
    else:
        batch_size = 300
    msg_box('SUITE2P REGISTRATION', 'Please select directory containing one recording (tif) file\n' 
                                    'Data file name must include "recording", otherwise it cannot be found', '+')
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    rec_dir = askdirectory()
    if rec_dir:
        suite2p_registering(rec_path=rec_dir, f_batch_size=batch_size)
    else:
        print('No Directory Selected')
