import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import os.path as osp
import platform
import sys

eval_set = 'val'
methods  = ['iclr_pos_ft_big_baseline_bn', 'iclr_gcn_dwr_pos_ft_spatial', 'iclr_internet_dwr_pos_ft_spatial', 'iclr_pos_ft_big_mask_common_spatial']
names = ['Factored3D', 'GCN', 'Interaction Net', '  Ours']
metrices = ['all', 'box2d+trans', 'box2d+scale', 'box2d+rot+shape']
suffixes = ['_False', '_True',  '_True',  '_True']

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', '..', 'cachedir')
plots_dir = os.path.join(cache_path, 'evaluation', 'icp', eval_set, 'plots')
import pdb

def subplots(plt, Y_X, sz_y_sz_x=(10,10)):
  Y,X = Y_X
  sz_y,sz_x = sz_y_sz_x
  plt.rcParams['figure.figsize'] = (X*sz_x, Y*sz_y)
  fig, axes = plt.subplots(Y, X)
  plt.subplots_adjust(wspace=0.1, hspace=0.1)
  return fig, axes





def pr_plots(net_names, suffixes, names, metric_index, metric_name, eval_set, set_number=0):
  dir_name = os.path.join(cache_path, 'evaluation', 'dwr')
  #json_file = os.path.join(dir_name, set_number, net_name, 'eval_set{}_0.json'.format(set_number))
  json_files = [os.path.join(dir_name, eval_set, net_name, 'eval_set{}_{}{}.json'.format(eval_set, set_number, suffix)) for (net_name, suffix) in zip(net_names, suffixes)]
 
  imsets = []
  datas = []
  for json_file in json_files:
      with open(json_file, 'rt') as f:
        a = json.load(f)
      datas.append(a)
      imset = a['eval_params']['set'].title()
      imsets.append(imset)

  plot_file = os.path.join('.', 'plots', 'eval_set{}.pdf'.format(metric_index))
  print('Saving plot to {}'.format(osp.abspath(plot_file)))
  # Plot 1 with AP for all, and minus other things one at a time.
  #with sns.axes_style("darkgrid"):
  with plt.style.context('fivethirtyeight'):
    fig, axes = subplots(plt, (1,1), (7,7))
    ax = axes
    legs = []
    i_order = [metric_index]
    # for i in np.arange(6, 12):
    # datas = datas[1:]
    for dx in range(len(datas)):
      a = datas[dx]
      for jx in [0]:
        i = i_order[jx]
        prec = np.array(a['bench_summary'][i]['prec'])
        rec = np.array(a['bench_summary'][i]['rec'])
       
        # label  = '{:4.1f} {:s}'.format(100*a['bench_summary'][i]['ap'], names[dx])
        if dx == 3:
          ax.plot(rec, prec, '-')
          legs.append('{:4.1f} {:s}'.format(100*a['bench_summary'][i]['ap'], names[dx]))
          # legs.append('{:4.1f} {:s}'.format(100*a['bench_summary'][i]['ap'], a['eval_params']['ap_str'][i]))
          # legs.append()
        else:
          ax.plot(rec, prec, '--')
          legs.append('{:4.1f}   {:s}'.format(100*a['bench_summary'][i]['ap'], names[dx]))
    
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1]);
    ax.set_xlabel('Recall', fontsize=20)
    ax.set_ylabel('Precision', fontsize=20)
    # ax.set_title('Precision Recall Plots for "{}"'.format(a['eval_params']['ap_str'][i]), fontsize=20)
    ax.set_title('PR Plots for "{}"'.format(metric_name), fontsize=20)

    handles, labels = ax.get_legend_handles_labels()

    # reverse the order
    # ax.legend(handles[::-1], labels[::-1])
    bbox_to_anchor = (0,0)
    l = ax.legend(legs, fontsize=18, loc='lower right' , framealpha=0.5, frameon=True)

    ax.plot([0,1], [0,0], 'k-')
    ax.plot([0,0], [0,1], 'k-')
    ax.axhline(linewidth=4, color="k")
    ax.axvline(linewidth=4, color="k")  
    plt.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='x', pad=15)
    ax.autoscale(enable=True, axis='both')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)

'''
  plot_file = os.path.join(dir_name, set_number, net_name, 'eval_set{}_0{}_frwd.pdf'.format(set_number, suffix))
  print('Saving plot to {}'.format(osp.abspath(plot_file)))

  with plt.style.context('fivethirtyeight'):
    fig, axes = subplots(plt, (1,1), (7,7))
    ax = axes
    legs = []
    i_order = [6, 9, 7, 8, 10, 11]
    # for i in np.arange(6, 12):
    for jx in range(6):
      i = i_order[jx]
      prec = np.array(a['bench_summary'][i]['prec'])
      rec = np.array(a['bench_summary'][i]['rec'])
      if i == 6:
        ax.plot(rec, prec, '-')
        legs.append('{:4.1f} {:s}'.format(100*a['bench_summary'][i]['ap'], a['eval_params']['ap_str'][i]))
      else:
        ax.plot(rec, prec, '--')
        str_ = '+'+'+'.join(a['eval_params']['ap_str'][i].split('+')[1:])
        legs.append('{:4.1f}   {:s}'.format(100*a['bench_summary'][i]['ap'], str_))
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1]);
    ax.set_xlabel('Recall', fontsize=20)
    ax.set_ylabel('Precision', fontsize=20)
    ax.set_title('Precision Recall Plots on {:s} Set'.format(imset), fontsize=20)

    l = ax.legend(legs, fontsize=18, bbox_to_anchor=(0,0), loc='lower left', framealpha=0.5, frameon=True)
    ax.plot([0,1], [0,0], 'k-')
    ax.plot([0,0], [0,1], 'k-')
    plt.tick_params(axis='both', which='major', labelsize=20)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)
'''

if __name__ == '__main__':
  pr_plots(methods, suffixes, names, 0, "All", eval_set)
  pr_plots(methods, suffixes, names, 8, "Box2D + Translation", eval_set)
  pr_plots(methods, suffixes, names, 10, "Box2D + Scale", eval_set)
  pr_plots(methods, suffixes, names, 7, "Box2D + Rot" , eval_set)
