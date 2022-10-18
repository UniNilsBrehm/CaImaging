import pandas as pd
import analysis_util_functions as uf
import NW_Detect_Stimulus as detect_stimulation
from IPython import embed

""" Convert all stimulus text files (from Nils Wenke's Experiments) in the selected directory the sutter MOM standard 
    and save it to HDD. Then detect all stimulus types (steps and ramps) in the stimulus voltage trace and store all
    parameters to a csv file.
    For finding the stimulation text files this script looks for the suffix ('tactile.txt') in the file names.
    Files that will be stored to HDD for every recording file in the directory:
    ..._protocol.csv
    ..._stimulation.txt
    
    File Names from Nils Wenke:
    180516_isl2bGcamp6s_4dpf_nr2_2_tactile.txt
    results_180512DOB_180516_2_2.txt
    STD_180516_isl2BGCamp6s_4dpf_nr2_2_stimulus_ROI.tif
    -----------------
    Nils Brehm - 2022
"""

# rename_files = False
# Select Directory and get all files
rec_dir = uf.select_dir()
rename_files = input('DO YOU WANT TO RENAME THE FILES (y/n): ')
if rename_files == 'y':
    # RENAME FILES
    uf.msg_box('INFO', f'SELECTED DATA: {rec_dir}', sep='+')
    n0, f0 = uf.rename_nw_stimulus_files(b_dir=rec_dir, do_copy=True, do_remove=True)
    n1, f1 = uf.rename_nw_result_files(b_dir=rec_dir, do_copy=True, do_remove=True)
    n2, f2 = uf.rename_nw_ref_files(b_dir=rec_dir, do_copy=True, do_remove=True)
    uf.msg_box('INFO', 'Renamed all files successfully!', sep='+')


stimulus_list = uf.get_files_by_suffix(
    rec_dir, tags=['tactile.txt'])
stimulus_list = stimulus_list['tactile.txt']
# STIMULUS CONVERTING AND DETECTING ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for rec_nr, v in enumerate(stimulus_list):
    rec_name = stimulus_list[rec_nr][:10]
    file_name = stimulus_list[rec_nr]
    uf.msg_box(f_header='INFO', f_msg=f'{rec_nr+1}/{len(stimulus_list)} - {rec_name}: STARTING DETECTION', sep='+')

    # Import Wenke stimulation file
    nw_stimulus = pd.read_csv(f'{rec_dir}/{file_name}', sep='\t', index_col=0, header=None)
    # Convert it to Sutter Standard
    stimulus_converted = detect_stimulation.convert_nw_stimulation_file(s_file=nw_stimulus)
    # Detect Stimulus from voltage trace
    stimulus, protocol = detect_stimulation.detect_stimuli(s_values=stimulus_converted)
    # Export it to HDD
    detect_stimulation.export_stimulus_file(
        s_file=stimulus,
        s_protocol=protocol,
        export_protocol_name=f'{rec_dir}/{rec_name}_protocol.csv',
        export_stimulus_name=f'{rec_dir}/{rec_name}_stimulation.txt'
    )
    uf.msg_box(f_header='INFO', f_msg=f'{rec_nr+1}/{len(stimulus_list)} - {rec_name}: Stimulus and Protocol files stored to HDD!', sep='+')
