import os


def create_directories(f_base_path):
    # Create directory
    dir_names = ['logs', 'original', 'rawdata', 'references', 'stimulation', 'tiffs']
    dir_names_logs = ['protocols', 'statics']
    for kk in dir_names:
        try:
            # Create target Directory
            os.mkdir(f'{f_base_path}/{kk}')
            print("Directory ", kk, " Created ")
        except FileExistsError:
            print(f'Directory: {kk} already exists!')
    try:
        # Create target Directory
        os.mkdir(f'{f_base_path}/logs/{dir_names_logs[0]}')
        os.mkdir(f'{f_base_path}/logs/{dir_names_logs[1]}')
    except FileExistsError:
        print(f'Directory: {dir_names_logs[0]} already exists!')
        print(f'Directory: {dir_names_logs[1]} already exists!')


# MAIN PART
b_dir = input('Enter base directory: ')
create_directories(b_dir)
print('DONE')
