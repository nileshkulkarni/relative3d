// data_stats = load('/home/nileshk/Research3/3dRelnet/relative3d/nnutils/../cachedir/snapshots/box3d_base_crf_potentials_stats.mat')
data_stats = load('..\\..\\..\\cachedir\\snapshots\\box3d_base_crf_potentials_stats.mat')
fields = ['trans', 'scale', 'quat', 'relative_trans', 'relative_scale', 'relative_dir_quant']
fitGMM(data_stats.relative_trans.cls_3_4)
