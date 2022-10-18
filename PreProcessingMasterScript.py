import create_dirs as cdir
import Detect_Stimulus as dt
import suite2p_pre_processing as s2p
import analysis_util_functions as uf
from tkinter.filedialog import askdirectory
from tkinter import Tk
import sys

# Get additional arguments from command line
uf.msg_box(
    'INFO',
    'You can give extra arguments to run only specific parts:\n --reg : Registration '
    '\n --rename : Renaming Files and metadata \n --detect: Stimulus Detection'
    , sep='+'
)

do_reg = False
do_rename = False
do_detection = False
if len(sys.argv) > 1:
    command = sys.argv[1]

    if command == '--reg':
        do_reg = True
    if command == '--rename':
        do_rename = True
    if command == '--detect':
        do_detection = True
else:
    do_reg = True
    do_rename = True
    do_detection = True

# Ask to select directory of data files
Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
rec_path = askdirectory()

# Take Care of data files structure
if do_rename:
    print('Do Rename')
    cdir.write_metadata(rec_path)
    cdir.create_directories(rec_path)
    cdir.rename_and_move_files(rec_path)

# Detect Stimulus Parameters from Voltage Trace
if do_detection:
    print('Do Detection')
    dt.detect_stimuli(rec_path, f_select_single_file=False)

# Registering recording (tiff) using suite2p software
if do_reg:
    print('Do Reg')
    s2p.suite2p_registering(rec_path)

uf.msg_box(f_header='INFO', f_msg='Master Script Finished!', sep='+')
uf.msg_box(f_header='INFO', f_msg='Now you have to draw rois ... Have Fun!', sep='+')

