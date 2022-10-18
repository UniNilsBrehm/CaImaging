import analysis_util_functions as uf

rec_dir = uf.select_dir()
uf.msg_box('INFO', f'SELECTED DATA: {rec_dir}', sep='+')

n0, f0 = uf.rename_nw_stimulus_files(b_dir=rec_dir, do_copy=True, do_remove=True)
n1, f1 = uf.rename_nw_result_files(b_dir=rec_dir, do_copy=True, do_remove=True)
n2, f2 = uf.rename_nw_ref_files(b_dir=rec_dir, do_copy=True, do_remove=True)

uf.msg_box('INFO', 'Renamed all files successfully!', sep='+')
