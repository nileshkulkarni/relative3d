"""Script for making html from a directory.
"""
# Sample usage:
# (box3d_shape_ft) python make_html.py --imgs_root_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/box3d/val/box3d_shape_ft' --html_name=box3d_shape_ft --html_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/pages/'

# (dwr_shape_ft) python make_html.py --imgs_root_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/dwr/val/dwr_shape_ft' --html_name=dwr_shape_ft --html_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/pages/'

# (depth_baseline) python make_html.py --imgs_root_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/depth_baseline' --html_name=depth_baseline --html_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/pages/'

# (voxels_baseline) python make_html.py --imgs_root_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/voxels_baseline' --html_name=voxels_baseline --html_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/pages/'

# (nyu) python make_html.py --imgs_root_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/nyu/test/dwr_shape_ft' --html_name=nyu_dwr_shape_ft --html_dir='/data0/shubhtuls/code/oc3d/cachedir/results_vis/pages/'

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
import os
import os.path as osp
from yattag import Doc
from yattag import indent
import numpy as np
import json
import pdb
flags.DEFINE_string('baseline_dir', '', 'Directory where renderings are saved')
flags.DEFINE_string('ours_dir', '', 'Directory where renderings are saved')
flags.DEFINE_string('vis_dirs', None, 'File names to visualize')
flags.DEFINE_string('out_dir', 'paper_vis', 'Directory to store results')

def main(_):

    opts = flags.FLAGS
    if opts.vis_dirs is None:
        assert False, 'required flag vis dirs'

    vis_dir_names = []
    with open(opts.vis_dirs) as f:
        for line in f:
            vis_dir_names.append(line.strip())


    base_image  = 'img.png'
    pred_image = 'b_pred_objects_cam_view.png'
    gt_image = 'c_gt_objects_cam_view.png'
    for i, vis_dir in enumerate(vis_dir_names):
        ## Copy the image
        baseline_image = osp.join(opts.baseline_dir, vis_dir, pred_image)
        ours_image = osp.join(opts.ours_dir, vis_dir, pred_image)
        rgb_image = osp.join(opts.baseline_dir, vis_dir, base_image)
        gt_3d_image = osp.join(opts.baseline_dir, vis_dir, gt_image)
        out_dir = opts.out_dir
        os.system('cp {} {}'.format(rgb_image, osp.join(out_dir, "{}_img.png".format(i))))
        os.system('cp {} {}'.format(gt_3d_image, osp.join(out_dir, "{}_gt.png".format(i))))
        os.system('cp {} {}'.format(baseline_image, osp.join(out_dir, "{}_baseline.png".format(i))))
        os.system('cp {} {}'.format(ours_image, osp.join(out_dir, "{}_ours.png".format(i))))


if __name__ == '__main__':
    app.run(main)
