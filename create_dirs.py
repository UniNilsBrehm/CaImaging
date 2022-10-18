import os
import pandas as pd
import shutil


def wenke_create_directories(f_base_path):
    # Create directory
    dir_names = ['01', '02', '03', '04', '05', '06', '07', '01_2', '02_2', '03_2', '04_2', '05_2']
    for kk in dir_names:
        try:
            # Create target Directory
            os.mkdir(f'{f_base_path}/{kk}')
            print("Directory ", kk, " Created ")
        except FileExistsError:
            print(f'Directory: {kk} already exists!')



def create_directories(f_base_path):
    # Create directory
    dir_names = ['protocol', 'original', 'recording', 'stimulation', 'figs', 'reg']
    for kk in dir_names:
        try:
            # Create target Directory
            os.mkdir(f'{f_base_path}/{kk}')
            print("Directory ", kk, " Created ")
        except FileExistsError:
            print(f'Directory: {kk} already exists!')


# def create_directories(f_base_path):
#     # Create directory
#     dir_names = ['logs', 'original', 'rawdata', 'references', 'stimulation', 'tiffs', 'stacks', 'figs', 'reg']
#     dir_names_logs = ['protocols', 'statics']
#     for kk in dir_names:
#         try:
#             # Create target Directory
#             os.mkdir(f'{f_base_path}/{kk}')
#             print("Directory ", kk, " Created ")
#         except FileExistsError:
#             print(f'Directory: {kk} already exists!')
#     try:
#         # Create target Directory
#         os.mkdir(f'{f_base_path}/logs/{dir_names_logs[0]}')
#         os.mkdir(f'{f_base_path}/logs/{dir_names_logs[1]}')
#         os.mkdir(f'{f_base_path}/original/sweeps')
#
#     except FileExistsError:
#         print(f'Directory: {dir_names_logs[0]} already exists!')
#         print(f'Directory: {dir_names_logs[1]} already exists!')


def write_metadata(f_base_path):
    # Look what files are in the directory and put the names all into one text file (metadata.txt)
    f_list = os.listdir(f_base_path)
    print('Collecting Metadata ....')
    metadata_df = pd.DataFrame()
    metadata_df['Stack'] = ['None']
    metadata_df['Recording'] = ['None']
    metadata_df['Stimulation'] = ['None']
    for f_name in f_list:
        if f_name.endswith('stack.TIF'):
            # Stack File
            # Make metadata entry
            metadata_df['Stack'] = [f_name]
        elif f_name.endswith('stimulus.TIF'):
            # Recording File
            # Make metadata entry
            metadata_df['Recording'] = [f_name]
        elif f_name.endswith('tactile.txt'):
            # Stimulus File
            # Make metadata entry
            metadata_df['Stimulation'] = [f_name]

    # Store metadata to HDD
    print(metadata_df)
    metadata_df.to_csv(f'{f_base_path}/metadata.csv', index=False)
    print('... successfully stored metadata to HDD')


def rename_and_move_files(f_base_path):
    stack_file, rec_file, stimulus_file = False, False, False
    f_list = os.listdir(f_base_path)
    print('Renaming and copying files ...')
    for f_name in f_list:
        if f_name.endswith('stack.TIF'):
            # Stack File
            # Copy and rename file
            src = f'{f_base_path}/{f_name}'
            shutil.copyfile(src, f'{f_base_path}/recording/stack.TIF')
            os.remove(src)
            stack_file = True
        elif f_name.endswith('stimulus.TIF'):
            # Recording File
            # Copy and rename file
            src = f'{f_base_path}/{f_name}'
            shutil.copyfile(src, f'{f_base_path}/original/original.TIF')
            os.remove(src)
            rec_file = True

        elif f_name.endswith('tactile.txt'):
            # Stimulus File
            # Copy and rename file
            src = f'{f_base_path}/{f_name}'
            shutil.copyfile(src, f'{f_base_path}/stimulation/stimulation_data.txt')
            os.remove(src)
            stimulus_file = True

    n_types = ['stack', 'recording', 'stimulation']

    for kk, vv in enumerate([stack_file, rec_file, stimulus_file]):
        if not vv:
            if n_types[kk] == 'stack':
                print(f'INFO: Could not find {n_types[kk]} file.')
            else:
                print(f'ERROR: Could not find {n_types[kk]} file !!!')

    if all([rec_file, stimulus_file]):
        print('... successful!')


if __name__ == '__main__':
    # MAIN PART
    b_dir = input('Enter base directory: ')
    # write_metadata(b_dir)
    # create_directories(b_dir)
    # rename_and_move_files(b_dir)
    wenke_create_directories(b_dir)